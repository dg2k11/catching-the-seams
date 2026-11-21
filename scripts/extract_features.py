#!/usr/bin/env python3
"""Step 2: Fit ReferenceBigramLM on disjoint Human NYT articles and cache standardized features."""

import argparse
import json
import pathlib
import pickle
import numpy as np
from seam_detector.config_util import load_yaml_config
from seam_detector.features.line_features import ReferenceBigramLM, extract_line_features
from seam_detector.features.seam_features import extract_seam_features


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_std(matrix):
    std = np.std(matrix, axis=0)
    return np.where(std < 1e-6, 1.0, std)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=None)
    parser.add_argument("--data-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    data_dir = args.data_dir or pathlib.Path(cfg.get("paths", {}).get("processed_dir", "data/processed"))

    ref_docs = load_jsonl(data_dir / "ref_lm_corpus.jsonl")
    ref_lm = ReferenceBigramLM()
    ref_lm.fit([d["sentences"] for d in ref_docs])

    raw_cache = {}
    for split_name in ("train", "validation", "test", "test_ood_noisy"):
        items = []
        for p in load_jsonl(data_dir / f"{split_name}.jsonl"):
            sents = [ln["text"] for ln in p["lines"]]
            labels = [ln["label"] for ln in p["lines"]]
            phi = extract_line_features(sents, ref_lm)
            psi = extract_seam_features(sents, phi)
            items.append({**p, "phi": phi, "psi": psi, "labels": labels})
        raw_cache[split_name] = items

    all_train_phi = np.vstack([it["phi"] for it in raw_cache["train"]])
    all_train_psi = np.vstack([it["psi"][1:] for it in raw_cache["train"] if len(it["psi"]) > 1])
    phi_mean = np.mean(all_train_phi, axis=0)
    phi_std = safe_std(all_train_phi)
    psi_mean = np.mean(all_train_psi, axis=0)
    psi_std = safe_std(all_train_psi)

    for items in raw_cache.values():
        for it in items:
            it["norm_phi"] = (it["phi"] - phi_mean) / phi_std
            norm_psi = (it["psi"] - psi_mean) / psi_std
            norm_psi[0] = 0.0
            it["norm_psi"] = norm_psi

    out_pkl = data_dir / "features_cache.pkl"
    with out_pkl.open("wb") as fh:
        pickle.dump(
            {
                "splits": raw_cache,
                "phi_mean": phi_mean,
                "phi_std": phi_std,
                "psi_mean": psi_mean,
                "psi_std": psi_std,
            },
            fh,
        )
    print(f"Saved feature cache (ref_lm fit on {len(ref_docs)} disjoint NYT Human articles) -> {out_pkl}")


if __name__ == "__main__":
    main()
