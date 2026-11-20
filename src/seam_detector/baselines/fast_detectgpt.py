# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""Baseline B1: Per-sentence Fast-DetectGPT curvature threshold (Bao et al., 2024)."""


class FastDetectGPTBaseline:
    """Threshold per-sentence conditional probability curvature tuned on validation."""

    def fit_threshold(self, curvatures, labels) -> float:
        raise NotImplementedError("TODO (Lead 2 — Baselines): tune curvature threshold on validation split.")

    def predict(self, curvatures):
        raise NotImplementedError("TODO (Lead 2 — Baselines): predict Human/AI labels from curvature.")
