# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""Interfaces for constructing Slice H and Slice E(k) passages (Proposal Section 3 & Appendix A.1)."""

from seam_detector.schema import Passage


def segment_into_passages(article_text: str, min_sentences: int = 6, max_sentences: int = 20) -> list[list[str]]:
    """Segment raw article narratives into 6-20 sentence windows (150-400 words)."""
    raise NotImplementedError("TODO (Lead 1 — Benchmark): implement sentence segmentation and length filtering.")


def build_slice_h(records: list[dict], num_passages: int = 200, seed: int = 221) -> list[Passage]:
    """Construct pure-human Slice H passages for measuring false-positive rate."""
    raise NotImplementedError("TODO (Lead 1 — Benchmark): implement Slice H sampling from Human_story articles.")


def build_slice_ek(
    records: list[dict],
    edit_rates: tuple[int, ...] = (0, 25, 50, 75),
    num_passages: int = 400,
    seed: int = 221,
) -> list[Passage]:
    """Construct hybrid Slice E(k) passages with ground-truth line-level edit logs."""
    raise NotImplementedError("TODO (Lead 1 — Benchmark): implement Slice E(k) hybrid passage generation.")
