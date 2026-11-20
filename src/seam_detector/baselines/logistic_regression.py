# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""Baseline B3 (main comparison): Independent per-line logistic regression on phi(c_t) via SGD."""

import numpy as np


class PerLineLogisticRegression:
    """Classifies each line independently using phi(c_t) without sequential state."""

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError("TODO (Lead 2 — Baselines): implement SGD logistic regression on phi(c_t).")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO (Lead 2 — Baselines): return P(y_t = AI | phi(c_t)).")
