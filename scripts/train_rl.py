#!/usr/bin/env python3
"""Step 4: Train Linear Q-learning, on-policy SARSA, and E2 state-ablation variants."""

import argparse
import pathlib
import pickle
import numpy as np
from seam_detector.config_util import load_yaml_config
from seam_detector.eval.metrics import compute_macro_f1
from seam_detector.features.state_featurizer import state_feature_names
from seam_detector.rl.env import SeamEpisodeEnv
from seam_detector.rl.q_learning import LinearRLAgent
from seam_detector.rl.sarsa import LinearSarsaAgent


def eval_agent_macro_f1(agent, items, include_seam=True, include_prev_label=True):
    all_y, all_p = [], []
    for it in items:
        env = SeamEpisodeEnv(
            it["norm_phi"],
            it["norm_psi"],
            it["labels"],
            include_seam=include_seam,
            include_prev_label=include_prev_label,
        )
        s = env.reset()
        done = False
        while not done:
            a = agent.act(s, epsilon=0.0)
            s, _, done = env.step(a)
        all_y.extend(it["labels"])
        all_p.extend(env.preds)
    return compute_macro_f1(all_y, all_p)


def train_variant(
    train_items,
    val_items,
    algorithm="q_learning",
    include_seam=True,
    include_prev_label=True,
    epochs=30,
    lr=0.006,
    discount=0.9,
    l2=1e-4,
    terminal_bonus_scale=2.0,
    seed=221,
):
    state_dim = len(state_feature_names())
    cls = LinearSarsaAgent if algorithm == "sarsa" else LinearRLAgent
    agent = cls(state_dim=state_dim, lr=lr, discount=discount, l2=l2, seed=seed)
    best_w = agent.w.copy()
    best_val = -1.0
    curve = []
    rng = np.random.default_rng(seed)
    decay_steps = epochs - 1 if epochs > 1 else 1

    for ep in range(epochs):
        eps = 0.30 - (0.25 * ep / decay_steps)
        cur_lr = lr / (1.0 + 0.04 * ep)
        for idx in rng.permutation(len(train_items)).tolist():
            it = train_items[idx]
            env = SeamEpisodeEnv(
                it["norm_phi"],
                it["norm_psi"],
                it["labels"],
                terminal_bonus_scale=terminal_bonus_scale,
                include_seam=include_seam,
                include_prev_label=include_prev_label,
            )
            s = env.reset()
            if algorithm == "q_learning":
                done = False
                while not done:
                    a = agent.act(s, epsilon=eps)
                    s_next, r, done = env.step(a)
                    agent.update(s, a, r, s_next, step_lr=cur_lr)
                    s = s_next
            else:
                a = agent.act(s, epsilon=eps)
                done = False
                while not done:
                    s_next, r, done = env.step(a)
                    a_next = None if done else agent.act(s_next, epsilon=eps)
                    agent.update_sarsa(s, a, r, s_next, a_next, step_lr=cur_lr)
                    s, a = s_next, a_next

        val_f1 = eval_agent_macro_f1(agent, val_items, include_seam, include_prev_label)
        curve.append(round(val_f1, 4))
        if val_f1 > best_val:
            best_val = val_f1
            best_w = agent.w.copy()

    agent.w = best_w
    return agent, curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=None)
    parser.add_argument("--data-dir", type=pathlib.Path, default=None)
    parser.add_argument("--ckpt-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = cfg.get("paths", {})
    tcfg = cfg.get("training", {})
    mcfg = cfg.get("mdp", {})
    data_dir = args.data_dir or pathlib.Path(paths.get("processed_dir", "data/processed"))
    ckpt_dir = args.ckpt_dir or pathlib.Path(paths.get("checkpoint_dir", "checkpoints"))

    epochs = tcfg.get("num_epochs", 30)
    lr = tcfg.get("learning_rate", 0.006)
    l2 = tcfg.get("l2", 1e-4)
    seed = cfg.get("seed", 221)
    bonus = mcfg.get("terminal_bonus_scale", 2.0)
    gammas = mcfg.get("discount_candidates", [0.9, 1.0])

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache = pickle.loads((data_dir / "features_cache.pkl").read_bytes())
    tr = cache["splits"]["train"]
    va = cache["splits"]["validation"]

    best_gamma = gammas[0]
    best_q_agent = None
    best_q_curve = None
    best_q_val = -1.0
    for gamma in gammas:
        ag, cv = train_variant(tr, va, "q_learning", True, True, epochs, lr, gamma, l2, bonus, seed)
        if max(cv) > best_q_val:
            best_q_val = max(cv)
            best_gamma = gamma
            best_q_agent = ag
            best_q_curve = cv

    sarsa_agent, sarsa_curve = train_variant(tr, va, "sarsa", True, True, epochs, lr, best_gamma, l2, bonus, seed)
    q_no_prev, _ = train_variant(tr, va, "q_learning", True, False, epochs, lr, best_gamma, l2, bonus, seed)
    q_no_seam, _ = train_variant(tr, va, "q_learning", False, True, epochs, lr, best_gamma, l2, bonus, seed)
    q_phi_only, _ = train_variant(tr, va, "q_learning", False, False, epochs, lr, best_gamma, l2, bonus, seed)

    out_path = ckpt_dir / "rl_agents.pkl"
    with out_path.open("wb") as fh:
        pickle.dump(
            {
                "best_gamma": best_gamma,
                "q_full": best_q_agent,
                "sarsa_full": sarsa_agent,
                "q_no_prev": q_no_prev,
                "q_no_seam": q_no_seam,
                "q_phi_only": q_phi_only,
                "q_val_curve": best_q_curve,
                "sarsa_val_curve": sarsa_curve,
            },
            fh,
        )
    print(f"Trained RL agents (best val gamma={best_gamma}, Q val Macro-F1={best_q_val:.4f}, SARSA val={max(sarsa_curve):.4f})")


if __name__ == "__main__":
    main()
