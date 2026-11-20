# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""Exact additive feature decompositions and E3 faithfulness evaluation (Proposal Sections 2 & 4)."""

import numpy as np
from seam_detector.schema import LinePrediction


def explain_line_decision(
    state_vec: np.ndarray,
    weights_human: np.ndarray,
    weights_ai: np.ndarray,
    feature_names: list[str],
    line_index: int,
    top_k: int = 4,
) -> LinePrediction:
    """Decompose Delta Q_t = (w_AI - w_H)^T f(s_t) into per-feature contributions."""
    raise NotImplementedError("TODO (Lead 4 — Explanations): implement exact linear margin decomposition.")


def evaluate_faithfulness_flip_rate(
    states: np.ndarray,
    weights_human: np.ndarray,
    weights_ai: np.ndarray,
    seed: int = 221,
) -> dict[str, float]:
    """Compare label flip rate when zeroing argmax_k |(w_AI,k - w_H,k) f_k(s_t)| vs. a random feature."""
    raise NotImplementedError("TODO (Lead 4 — Explanations): implement E3 faithfulness flip-rate test.")
