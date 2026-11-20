# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""CLI entrypoint for building Slice H and Slice E(k) from Rajarshi-Roy-research/Defactify_Text_Dataset."""

import argparse
import pathlib
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="building Slice H and Slice E(k) from Rajarshi-Roy-research/Defactify_Text_Dataset")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/default.yaml"))
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    raise NotImplementedError("TODO (Lead 1 — Benchmark): wire build_benchmark.py with config keys: " + ", ".join(cfg.keys()))


if __name__ == "__main__":
    main()
