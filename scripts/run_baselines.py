#!/usr/bin/env python3
"""Step 3: Fit Baseline B1 (bigram smoothness) and Baseline B3 (per-line SGD logistic regression)."""

import argparse
import pathlib
import pickle
import numpy as np
from seam_detector.baselines.fast_detectgpt import BigramSmoothnessBaseline
from seam_detector.baselines.logistic_regression import PerLineLogisticRegression
from seam_detector.config_util import load_yaml_config


def stack_split(items):
    x = np.vstack([it["norm_phi"] for it in items])
    smooth = np.concatenate([it["phi"][:, 1] for it in items])
    y = np.concatenate([np.array(it["labels"], dtype=int) for it in items])
    return x, smooth, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=None)
    parser.add_argument("--data-dir", type=pathlib.Path, default=None)
    parser.add_argument("--ckpt-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = cfg.get("paths", {})
    bcfg = cfg.get("baselines", {})
    data_dir = args.data_dir or pathlib.Path(paths.get("processed_dir", "data/processed"))
    ckpt_dir = args.ckpt_dir or pathlib.Path(paths.get("checkpoint_dir", "checkpoints"))

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache = pickle.loads((data_dir / "features_cache.pkl").read_bytes())
    x_tr, _, y_tr = stack_split(cache["splits"]["train"])
    x_va, sm_va, y_va = stack_split(cache["splits"]["validation"])

    b1 = BigramSmoothnessBaseline()
    thr_b1, dir_b1 = b1.fit(sm_va, y_va)

    b3 = PerLineLogisticRegression(
        lr=bcfg.get("b3_lr", 0.06),
        epochs=bcfg.get("b3_epochs", 40),
        l2=bcfg.get("b3_l2", 1e-3),
        seed=cfg.get("seed", 221),
    )
    b3.fit(x_tr, y_tr, x_va, y_va)

    out_path = ckpt_dir / "baselines.pkl"
    with out_path.open("wb") as fh:
        pickle.dump({"b1": b1, "b3": b3}, fh)
    print(f"Trained baselines -> {out_path} (B1 {dir_b1} thr={thr_b1:.3f}, B3 val prob thr={b3.threshold:.2f})")


if __name__ == "__main__":
    main()
