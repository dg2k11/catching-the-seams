# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""Baseline B2: Zero-shot Gemini line-by-line provenance prompting (evaluation only)."""

from seam_detector.schema import Label, Passage


def predict_passage_zeroshot(passage: Passage, model_name: str = "gemini-2.5-flash") -> list[Label]:
    """Prompt an LLM zero-shot to tag each sentence in a passage as Human or AI."""
    raise NotImplementedError("TODO (Lead 2 — Baselines): implement zero-shot line tagging baseline B2.")
