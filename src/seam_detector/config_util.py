"""YAML configuration loader for pipeline CLI scripts."""

import pathlib
import yaml


def load_yaml_config(config_path):
    if config_path is None:
        return {}
    path = pathlib.Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
