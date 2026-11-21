"""Baseline B1: single-feature bigram smoothness threshold tuned on validation."""

import numpy as np
from seam_detector.eval.metrics import compute_macro_f1


class BigramSmoothnessBaseline:
    def __init__(self):
        self.threshold = 0.0
        self.direction = "ge"

    def fit(self, scores, y_val):
        best_thr = 0.0
        best_dir = "ge"
        best_f1 = -1.0
        y_list = y_val.tolist()
        for direction in ("ge", "le"):
            for q in np.linspace(5, 95, 91):
                thr = float(np.percentile(scores, q))
                preds = (scores >= thr).astype(int) if direction == "ge" else (scores <= thr).astype(int)
                f1 = compute_macro_f1(y_list, preds.tolist())
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr
                    best_dir = direction
        self.threshold = best_thr
        self.direction = best_dir
        return best_thr, best_dir

    def predict(self, scores):
        if self.direction == "ge":
            return (scores >= self.threshold).astype(int)
        return (scores <= self.threshold).astype(int)
