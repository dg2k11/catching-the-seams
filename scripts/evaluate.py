#!/usr/bin/env python3
"""Step 5: Evaluate E1 robustness, E2 state ablations, E3 faithfulness, and walkthrough."""

import argparse
import json
import pathlib
import pickle
from seam_detector.baselines.oracle import edit_log_oracle, oracle_neighbor_rollout
from seam_detector.config_util import load_yaml_config
from seam_detector.eval.explanations import evaluate_faithfulness_flip_rate, explain_line_decision
from seam_detector.eval.metrics import compute_macro_f1, compute_passage_share_mae, compute_slice_h_fpr
from seam_detector.features.state_featurizer import build_state_vector, state_feature_names


def collect_predictions(items, predictor_fn):
    by_slice = {}
    true_shares_e, pred_shares_e = [], []
    all_e_y, all_e_p = [], []

    for it in items:
        preds = predictor_fn(it)
        labels = it["labels"]
        sl = it["slice_name"]
        by_slice.setdefault(sl, {"y": [], "p": []})
        by_slice[sl]["y"].extend(labels)
        by_slice[sl]["p"].extend(preds)
        if sl != "H":
            true_shares_e.append(sum(labels) / len(labels))
            pred_shares_e.append(sum(preds) / len(preds))
            all_e_y.extend(labels)
            all_e_p.extend(preds)

    unedited_y = by_slice["H"]["y"] + by_slice["E_0"]["y"]
    unedited_p = by_slice["H"]["p"] + by_slice["E_0"]["p"]
    e0_preds = by_slice["E_0"]["p"]
    e0_ai_recall = sum(1 for yp in e0_preds if yp == 1) / len(e0_preds) if e0_preds else 0.0

    return {
        "slice_h_fpr": round(compute_slice_h_fpr(by_slice["H"]["p"]), 4),
        "unedited_h_plus_e0_f1": round(compute_macro_f1(unedited_y, unedited_p), 4),
        "E_0_ai_recall": round(e0_ai_recall, 4),
        "E_25_f1": round(compute_macro_f1(by_slice["E_25"]["y"], by_slice["E_25"]["p"]), 4),
        "E_50_f1": round(compute_macro_f1(by_slice["E_50"]["y"], by_slice["E_50"]["p"]), 4),
        "E_75_f1": round(compute_macro_f1(by_slice["E_75"]["y"], by_slice["E_75"]["p"]), 4),
        "pooled_ek_f1": round(compute_macro_f1(all_e_y, all_e_p), 4),
        "passage_mae": round(compute_passage_share_mae(true_shares_e, pred_shares_e), 4),
    }


def rollout_rl(agent, it, include_seam=True, include_prev=True):
    preds = []
    states = []
    n = len(it["labels"])
    for t in range(n):
        prev_lbl = None if t == 0 else preds[t - 1]
        state = build_state_vector(
            it["norm_phi"][t],
            it["norm_psi"][t],
            prev_lbl,
            t / n,
            include_seam=include_seam,
            include_prev_label=include_prev,
        )
        states.append(state)
        preds.append(agent.act(state, epsilon=0.0))
    return preds, states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=None)
    parser.add_argument("--data-dir", type=pathlib.Path, default=None)
    parser.add_argument("--ckpt-dir", type=pathlib.Path, default=None)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = cfg.get("paths", {})
    data_dir = args.data_dir or pathlib.Path(paths.get("processed_dir", "data/processed"))
    ckpt_dir = args.ckpt_dir or pathlib.Path(paths.get("checkpoint_dir", "checkpoints"))
    out_dir = args.out_dir or pathlib.Path(paths.get("output_dir", "outputs"))
    seed = cfg.get("seed", 221)

    out_dir.mkdir(parents=True, exist_ok=True)
    cache = pickle.loads((data_dir / "features_cache.pkl").read_bytes())
    baselines = pickle.loads((ckpt_dir / "baselines.pkl").read_bytes())
    rl = pickle.loads((ckpt_dir / "rl_agents.pkl").read_bytes())
    test_items = cache["splits"]["test"]

    b1 = baselines["b1"]
    b3 = baselines["b3"]
    q_full = rl["q_full"]
    sarsa_full = rl["sarsa_full"]

    e1_results = {
        "B1_smoothness_proxy": collect_predictions(test_items, lambda it: b1.predict(it["phi"][:, 1]).tolist()),
        "B3_logistic_regression": collect_predictions(test_items, lambda it: b3.predict(it["norm_phi"]).tolist()),
        "Linear_SARSA": collect_predictions(test_items, lambda it: rollout_rl(sarsa_full, it)[0]),
        "Linear_Q_Learning": collect_predictions(test_items, lambda it: rollout_rl(q_full, it)[0]),
        "Oracle_Neighbor_Bound": collect_predictions(
            test_items,
            lambda it: oracle_neighbor_rollout(q_full, it["norm_phi"], it["norm_psi"], it["labels"]),
        ),
        "Edit_Log_Oracle": collect_predictions(test_items, lambda it: edit_log_oracle(it["labels"])),
    }

    e2_ablation = {
        "Full_State_phi_psi_prev": e1_results["Linear_Q_Learning"],
        "No_Prev_Label_phi_psi": collect_predictions(test_items, lambda it: rollout_rl(rl["q_no_prev"], it, True, False)[0]),
        "No_Seam_Shift_phi_prev": collect_predictions(test_items, lambda it: rollout_rl(rl["q_no_seam"], it, False, True)[0]),
        "Line_Only_phi": collect_predictions(test_items, lambda it: rollout_rl(rl["q_phi_only"], it, False, False)[0]),
    }

    all_test_states = []
    for it in test_items:
        _, sts = rollout_rl(q_full, it)
        all_test_states.extend(sts)
    e3_faithfulness = evaluate_faithfulness_flip_rate(all_test_states, q_full.w[0], q_full.w[1], seed=seed)

    # Select a short hybrid test passage (prioritizing E_25) where Q outperforms B3.
    feat_names = state_feature_names()
    walkthrough_case = None
    best_score = None
    for it in test_items:
        if it["slice_name"] not in ("E_25", "E_50", "E_75"):
            continue
        b3_p = b3.predict(it["norm_phi"]).tolist()
        b3_prob = b3.predict_proba(it["norm_phi"]).tolist()
        q_p, sts = rollout_rl(q_full, it)
        y = it["labels"]
        q_acc = sum(1 for a, b in zip(y, q_p) if a == b)
        b3_acc = sum(1 for a, b in zip(y, b3_p) if a == b)
        if q_acc >= len(y) - 2 and q_acc > b3_acc and len(y) <= 9:
            cand_score = (it["slice_name"] == "E_25", q_acc - b3_acc, q_acc, -len(y))
            if best_score is not None and cand_score <= best_score:
                continue
            best_score = cand_score
            lines_out = []
            for t in range(len(y)):
                exp = explain_line_decision(sts[t], q_full.w[0], q_full.w[1], feat_names, t, y[t], top_k=3)
                lines_out.append(
                    {
                        "t": t + 1,
                        "text": it["lines"][t]["text"],
                        "source_model": it["lines"][t]["source_model"],
                        "true_label": "AI" if y[t] == 1 else "Human",
                        "b3_pred": "AI" if b3_p[t] == 1 else "Human",
                        "b3_prob_ai": round(b3_prob[t], 2),
                        "q_pred": "AI" if exp.predicted_label == 1 else "Human",
                        "margin_delta_q": round(exp.margin_delta_q, 2),
                        "top_reasons": [(k, round(v, 2)) for k, v in exp.top_contributions],
                    }
                )
            walkthrough_case = {
                "passage_id": it["passage_id"],
                "num_lines": len(y),
                "q_correct": q_acc,
                "b3_correct": b3_acc,
                "true_ai_share": round(sum(y) / len(y), 4),
                "b3_ai_share": round(sum(b3_p) / len(y), 4),
                "q_ai_share": round(sum(q_p) / len(y), 4),
                "lines": lines_out,
            }

    ood_items = cache["splits"]["test_ood_noisy"]
    ood_comparison = {
        "B3_logistic_regression": collect_predictions(ood_items, lambda it: b3.predict(it["norm_phi"]).tolist()),
        "Linear_Q_Learning": collect_predictions(ood_items, lambda it: rollout_rl(q_full, it)[0]),
    }

    report_payload = {
        "best_gamma": rl["best_gamma"],
        "b1_threshold": round(b1.threshold, 4),
        "b1_direction": b1.direction,
        "b3_threshold": round(b3.threshold, 2),
        "q_val_curve": rl["q_val_curve"],
        "sarsa_val_curve": rl["sarsa_val_curve"],
        "e1_results": e1_results,
        "e2_ablation": e2_ablation,
        "e3_faithfulness": e3_faithfulness,
        "ood_noisy_test": ood_comparison,
        "walkthrough_case": walkthrough_case,
    }
    (out_dir / "evaluation_results.json").write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(json.dumps(report_payload, indent=2))


if __name__ == "__main__":
    main()
