#!/usr/bin/env python3
"""Step 1: Clean Defactify Parquet splits, partition by story_group, and build Slice H + Slice E(k)."""

import argparse
import json
import pathlib
import random
from seam_detector.config_util import load_yaml_config
from seam_detector.data.benchmark_builder import build_split_passages
from seam_detector.data.defactify_loader import load_cleaned_split


def serialize_passage(passage):
    return {
        "passage_id": passage.passage_id,
        "slice_name": passage.slice_name,
        "edit_rate": passage.edit_rate,
        "split": passage.split,
        "lines": [
            {
                "line_index": ln.line_index,
                "text": ln.text,
                "label": int(ln.label),
                "source_model": ln.source_model,
                "was_rewritten": ln.was_rewritten,
            }
            for ln in passage.lines
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=None)
    parser.add_argument("--parquet-dir", type=pathlib.Path, default=None)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = cfg.get("paths", {})
    dcfg = cfg.get("dataset", {})
    parquet_dir = args.parquet_dir or pathlib.Path(paths.get("parquet_dir", "/tmp/defactify_parquet"))
    out_dir = args.out_dir or pathlib.Path(paths.get("processed_dir", "data/processed"))
    seed = cfg.get("seed", 221)
    min_s = dcfg.get("min_sentences", 6)
    max_s = dcfg.get("max_sentences", 16)
    min_w = dcfg.get("min_words", 150)
    max_w = dcfg.get("max_words", 400)
    edit_rates = tuple(dcfg.get("edit_rates", [0, 25, 50, 75]))
    tr_h, tr_k = dcfg.get("train_slice_h", 210), dcfg.get("train_per_k", 105)
    ev_h, ev_k = dcfg.get("eval_slice_h", 45), dcfg.get("eval_per_k", 22)

    out_dir.mkdir(parents=True, exist_ok=True)
    tr_all, tr_stats = load_cleaned_split(parquet_dir / "train.parquet")
    va_all, va_stats = load_cleaned_split(parquet_dir / "val.parquet")
    ood_all, ood_stats = load_cleaned_split(parquet_dir / "test.parquet")

    # Exclude any training story_group whose sentences also appear in validation or noisy test.
    val_sents = {s for r in va_all for s in r["sentences"][:16]}
    ood_sents = {s for r in ood_all for s in r["sentences"][:16]}
    blocked_groups = {
        r["story_group"]
        for r in tr_all
        if any(s in val_sents or s in ood_sents for s in r["sentences"][:16])
    }
    tr_clean = [r for r in tr_all if r["story_group"] not in blocked_groups]

    story_groups = sorted({r["story_group"] for r in tr_clean})
    random.Random(seed).shuffle(story_groups)
    n_groups = len(story_groups)
    c1 = int(0.15 * n_groups)
    c2 = int(0.82 * n_groups)
    ref_groups = set(story_groups[:c1])
    train_groups = set(story_groups[c1:c2])
    test_groups = set(story_groups[c2:])

    ref_pool = [r for r in tr_clean if r["story_group"] in ref_groups and r["label_a"] == 0]
    train_pool = [r for r in tr_clean if r["story_group"] in train_groups]
    clean_test_pool = [r for r in tr_clean if r["story_group"] in test_groups]

    split_specs = [
        ("train", train_pool, {**tr_stats, "story_groups": len(train_groups), "pool_articles": len(train_pool)}, tr_h, tr_k, seed),
        ("validation", va_all, {**va_stats, "story_groups": len({r["story_group"] for r in va_all}), "pool_articles": len(va_all)}, ev_h, ev_k, seed + 1),
        ("test", clean_test_pool, {"story_groups": len(test_groups), "pool_articles": len(clean_test_pool)}, ev_h, ev_k, seed + 2),
        ("test_ood_noisy", ood_all, {**ood_stats, "pool_articles": len(ood_all)}, ev_h, ev_k, seed + 3),
    ]

    summary = {"blocked_cross_split_groups": len(blocked_groups), "ref_lm_human_articles": len(ref_pool)}
    used_split_sents = set()
    for split_name, pool, stats, n_h, n_k, sp_seed in split_specs:
        passages = build_split_passages(
            pool,
            split_name,
            n_slice_h=n_h,
            n_per_k=n_k,
            edit_rates=edit_rates,
            seed=sp_seed,
            min_s=min_s,
            max_s=max_s,
            min_words=min_w,
            max_words=max_w,
        )
        out_file = out_dir / f"{split_name}.jsonl"
        with out_file.open("w", encoding="utf-8") as fh:
            for p in passages:
                fh.write(json.dumps(serialize_passage(p)) + "\n")
                for ln in p.lines:
                    used_split_sents.add(ln.text.strip())
        total_lines = sum(len(p.lines) for p in passages)
        ai_lines = sum(sum(1 for ln in p.lines if ln.label == 1) for p in passages)
        wcs = [sum(len(ln.text.split()) for ln in p.lines) for p in passages]
        summary[split_name] = {
            **stats,
            "passages": len(passages),
            "slice_h_passages": n_h,
            "slice_ek_passages_per_k": n_k,
            "total_lines": total_lines,
            "human_lines": total_lines - ai_lines,
            "ai_lines": ai_lines,
            "min_words": min(wcs),
            "max_words": max(wcs),
            "min_sents": min(len(p.lines) for p in passages),
            "max_sents": max(len(p.lines) for p in passages),
        }
        print(f"[{split_name}] {len(passages)} passages, {total_lines} lines ({ai_lines} AI / {total_lines - ai_lines} Human), words=[{min(wcs)}, {max(wcs)}]")

    ref_out = out_dir / "ref_lm_corpus.jsonl"
    with ref_out.open("w", encoding="utf-8") as fh:
        for r in ref_pool:
            clean_sents = [s for s in r["sentences"] if s.strip() not in used_split_sents]
            if clean_sents:
                fh.write(json.dumps({"sentences": clean_sents}) + "\n")

    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
