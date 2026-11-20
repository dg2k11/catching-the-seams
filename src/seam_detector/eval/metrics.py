# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""Evaluation metrics (Proposal Section 3): Macro-F1 per k, Slice H FPR, |alpha_hat - alpha*|, faithfulness."""


def compute_macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    """Compute pooled binary Macro-F1 across Human (0) and AI (1) lines."""
    raise NotImplementedError("TODO (Lead 4 — Evaluation): implement pooled line-level Macro-F1.")


def compute_slice_h_fpr(y_pred_slice_h: list[int]) -> float:
    """Compute false-positive rate on pure-human Slice H lines."""
    raise NotImplementedError("TODO (Lead 4 — Evaluation): implement Slice H false-positive rate.")


def compute_passage_share_mae(true_shares: list[float], pred_shares: list[float]) -> float:
    """Compute mean |alpha_hat - alpha*| across passages."""
    raise NotImplementedError("TODO (Lead 4 — Evaluation): implement passage AI-share error.")
