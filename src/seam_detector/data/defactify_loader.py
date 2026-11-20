# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""Loader interface for Rajarshi-Roy-research/Defactify_Text_Dataset (arXiv:2510.22874)."""

from typing import Iterable


def load_defactify_split(split: str, hf_repo: str = "Rajarshi-Roy-research/Defactify_Text_Dataset") -> Iterable[dict]:
    """Stream or load a split ('train', 'validation', or 'test') from Hugging Face.

    Each record contains:
      - 'Text': full article text
      - 'Label_A': 0 (Human) or 1 (AI)
      - 'Label_B': 'Human_story' or one of the 6 source LLMs
    """
    raise NotImplementedError("TODO (Lead 1 — Benchmark): implement Defactify split loading and filtering.")
