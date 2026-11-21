"""Unit tests for schema containers, Defactify cleaning, and RL agents."""

import pathlib
import random
import unittest
import numpy as np
import yaml

from seam_detector.baselines.fast_detectgpt import BigramSmoothnessBaseline
from seam_detector.baselines.logistic_regression import PerLineLogisticRegression
from seam_detector.baselines.oracle import edit_log_oracle, oracle_neighbor_rollout
from seam_detector.data.benchmark_builder import crop_window
from seam_detector.data.defactify_loader import clean_article_to_sentences, is_corrupted_or_boilerplate
from seam_detector.eval.explanations import evaluate_faithfulness_flip_rate, explain_line_decision
from seam_detector.eval.metrics import compute_f1_for_class, compute_macro_f1, compute_passage_share_mae, compute_slice_h_fpr
from seam_detector.features.line_features import FEATURE_NAMES, ReferenceBigramLM, extract_line_features
from seam_detector.features.seam_features import SEAM_FEATURE_NAMES, char_3gram_cosine_dist, extract_seam_features
from seam_detector.features.state_featurizer import build_state_vector, state_feature_names
from seam_detector.rl.env import SeamEpisodeEnv
from seam_detector.rl.q_learning import LinearRLAgent
from seam_detector.rl.sarsa import LinearSarsaAgent
from seam_detector.schema import Label, LineRecord, Passage


SAMPLE_HUMAN_ARTICLES = [
    [
        "City budget officials met Tuesday morning to review transit funding numbers for next year.",
        "Several council members asked whether bus route changes would delay commuter trains.",
        "According to the report, ticket sales rose six percent between March and August.",
    ],
    [
        "County health workers opened three neighborhood clinics before winter vaccinations began.",
        "Local pharmacies reported steady turnout on Saturday despite heavy afternoon rain.",
    ],
]

SAMPLE_PASSAGE_SENTS = [
    "City budget officials met Tuesday morning to review transit funding numbers for next year.",
    "Furthermore, municipal agencies completed their quarterly evaluation of transit schedules.",
    "County auditors scheduled another public hearing for Thursday afternoon at City Hall.",
    "Ultimately, city planners approved construction funding for two new commuter stations.",
]


class RepoAndCoreTests(unittest.TestCase):
    def test_passage_ai_share(self):
        lines = [
            LineRecord(0, SAMPLE_PASSAGE_SENTS[0], Label.HUMAN),
            LineRecord(1, SAMPLE_PASSAGE_SENTS[1], Label.AI),
            LineRecord(2, SAMPLE_PASSAGE_SENTS[2], Label.HUMAN),
            LineRecord(3, SAMPLE_PASSAGE_SENTS[3], Label.AI),
        ]
        passage = Passage("demo_001", "E_50", 50, "validation", lines)
        self.assertEqual(passage.num_lines, 4)
        self.assertAlmostEqual(passage.true_ai_share, 0.50)

    def test_default_config_loads(self):
        cfg_path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self.assertEqual(cfg["dataset"]["hf_repo"], "Rajarshi-Roy-research/Defactify_Text_Dataset")
        self.assertEqual(cfg["dataset"]["edit_rates"], [0, 25, 50, 75])
        self.assertEqual(cfg["features"]["num_line_features"], 18)
        self.assertEqual(cfg["features"]["state_dim"], 32)
        self.assertEqual(cfg["mdp"]["actions"], ["Human", "AI"])

    def test_loader_cleaning_and_repair(self):
        raw = (
            "Supported by\n"
            "Climate|Wildfire Briefing\n"
            "## A Short Title\n"
            "Firefighters cleared fallen trees along Highway 9 before dusk.State crews inspected damaged bridges along Oak Creek after midnight.\n"
            "Local residents returned home on Wednesday after officials lifted the warning.\n"
        )
        sents = clean_article_to_sentences(raw)
        self.assertEqual(len(sents), 3)
        self.assertTrue(sents[0].startswith("Firefighters"))
        self.assertTrue(sents[1].startswith("State crews"))
        self.assertTrue(is_corrupted_or_boilerplate("Too short."))
        self.assertTrue(is_corrupted_or_boilerplate("Error communicating with upstream API " * 20))

    def test_crop_window_respects_bounds(self):
        base = " ".join(["word"] * 25) + "."
        sents = [f"Sentence {i} {base}" for i in range(12)]
        win = crop_window(sents, random.Random(221), min_s=6, max_s=16, min_words=150, max_words=400)
        self.assertIsNotNone(win)
        words = sum(len(s.split()) for s in win)
        self.assertTrue(6 <= len(win) <= 16)
        self.assertTrue(150 <= words <= 400)

    def test_feature_dimensions_and_seam_guards(self):
        lm = ReferenceBigramLM()
        lm.fit(SAMPLE_HUMAN_ARTICLES)
        phi = extract_line_features(SAMPLE_PASSAGE_SENTS, lm)
        psi = extract_seam_features(SAMPLE_PASSAGE_SENTS, phi)
        self.assertEqual(phi.shape, (len(SAMPLE_PASSAGE_SENTS), len(FEATURE_NAMES)))
        self.assertEqual(psi.shape, (len(SAMPLE_PASSAGE_SENTS), len(SEAM_FEATURE_NAMES)))
        self.assertTrue(np.allclose(psi[0], 0.0))
        self.assertAlmostEqual(char_3gram_cosine_dist("testing text", "testing text"), 0.0)
        self.assertAlmostEqual(char_3gram_cosine_dist("hi", "hi"), 1.0)

        names = state_feature_names()
        state = build_state_vector(phi[1], psi[1], prev_label=Label.HUMAN, rel_pos=0.25)
        self.assertEqual(len(names), 32)
        self.assertEqual(state.shape, (32,))
        self.assertEqual(state[-4], -1.0)
        self.assertEqual(state[-3], 0.0)
        self.assertEqual(state[-1], 1.0)

    def test_rl_agents_env_and_oracle(self):
        rng = np.random.default_rng(7)
        norm_phi = rng.normal(size=(4, 18))
        norm_psi = rng.normal(size=(4, 5))
        norm_psi[0] = 0.0
        labels = [0, 1, 0, 1]

        env = SeamEpisodeEnv(norm_phi, norm_psi, labels, terminal_bonus_scale=2.0)
        q_agent = LinearRLAgent(state_dim=32, lr=0.05, discount=0.9, seed=7)
        s = env.reset()
        done = False
        while not done:
            s_next, r, done = env.step(1)
            q_agent.update(s, 1, r, s_next)
            s = s_next
        self.assertGreater(np.linalg.norm(q_agent.w), 0.0)

        sarsa_agent = LinearSarsaAgent(state_dim=32, lr=0.05, discount=0.9, seed=7)
        env2 = SeamEpisodeEnv(norm_phi, norm_psi, labels)
        s = env2.reset()
        a = sarsa_agent.act(s, epsilon=0.0)
        s_next, r, done = env2.step(a)
        a_next = sarsa_agent.act(s_next, epsilon=0.0)
        err = sarsa_agent.update_sarsa(s, a, r, s_next, a_next)
        self.assertTrue(isinstance(err, float))
        self.assertGreater(np.linalg.norm(sarsa_agent.w), 0.0)

        self.assertEqual(edit_log_oracle(labels), labels)
        ora_preds = oracle_neighbor_rollout(q_agent, norm_phi, norm_psi, labels)
        self.assertEqual(len(ora_preds), 4)

    def test_baselines_metrics_and_explanations(self):
        x_tr = np.vstack([np.full((20, 18), -1.0), np.full((20, 18), 1.0)])
        y_tr = np.array([0] * 20 + [1] * 20)
        b3 = PerLineLogisticRegression(lr=0.1, epochs=15, l2=1e-4, seed=1)
        b3.fit(x_tr, y_tr, x_tr, y_tr)
        self.assertGreaterEqual(compute_macro_f1(y_tr.tolist(), b3.predict(x_tr).tolist()), 0.99)

        b1 = BigramSmoothnessBaseline()
        b1.fit(np.array([-1.0, -0.5, 0.5, 1.0]), np.array([0, 0, 1, 1]))
        self.assertEqual(b1.predict(np.array([-1.0, 1.0])).tolist(), [0, 1])

        self.assertEqual(compute_f1_for_class([], [], 1), 0.0)
        self.assertEqual(compute_f1_for_class([0, 0], [0, 0], 1), 1.0)
        self.assertAlmostEqual(compute_slice_h_fpr([0, 1, 0, 0]), 0.25)
        self.assertAlmostEqual(compute_passage_share_mae([0.5, 0.25], [0.75, 0.25]), 0.125)

        w_h = np.zeros(32)
        w_ai = np.zeros(32)
        w_ai[0] = 3.0
        s = np.zeros(32)
        s[0] = 2.0
        s[1] = 0.5
        s[-1] = 1.0
        pred = explain_line_decision(s, w_h, w_ai, state_feature_names(), line_index=0, ground_truth=1, top_k=2)
        self.assertEqual(pred.predicted_label, Label.AI)
        self.assertAlmostEqual(pred.margin_delta_q, 6.0)
        e3 = evaluate_faithfulness_flip_rate([s, s], w_h, w_ai, seed=1)
        self.assertEqual(e3["num_lines"], 2)


if __name__ == "__main__":
    unittest.main()
