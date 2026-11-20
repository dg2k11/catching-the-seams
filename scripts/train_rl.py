# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""CLI entrypoint for training linear Q-learning and SARSA agents in NumPy on CPU."""

import argparse
import pathlib
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="training linear Q-learning and SARSA agents in NumPy on CPU")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/default.yaml"))
    parser.add_argument("--algorithm", choices=["q_learning", "sarsa", "policy_eval"], default="q_learning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    raise NotImplementedError("TODO (Lead 3 — MDP & RL): wire train_rl.py with config keys: " + ", ".join(cfg.keys()))


if __name__ == "__main__":
    main()
