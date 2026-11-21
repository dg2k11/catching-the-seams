"""Baseline B3: per-line logistic regression on standardized phi(c_t)."""

import numpy as np
from seam_detector.eval.metrics import compute_macro_f1


class PerLineLogisticRegression:
    def __init__(self, lr=0.06, epochs=40, l2=1e-3, seed=221):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.w = None
        self.b = 0.0
        self.threshold = 0.5

    def fit(self, x_train, y_train, x_val, y_val):
        rng = np.random.default_rng(self.seed)
        n, d = x_train.shape
        self.w = np.zeros(d, dtype=np.float64)
        self.b = 0.0
        n_pos = max(1.0, float(np.sum(y_train == 1)))
        n_neg = max(1.0, float(np.sum(y_train == 0)))
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)

        batch_size = 128
        for ep in range(self.epochs):
            perm = rng.permutation(n)
            eta = self.lr / (1.0 + 0.05 * ep)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = x_train[idx]
                yb = y_train[idx]
                probs = 1.0 / (1.0 + np.exp(-np.clip(xb @ self.w + self.b, -30.0, 30.0)))
                sample_w = np.where(yb == 1, w_pos, w_neg)
                err = (probs - yb) * sample_w
                self.w -= eta * ((xb.T @ err) / len(idx) + self.l2 * self.w)
                self.b -= eta * float(np.mean(err))

        val_probs = self.predict_proba(x_val)
        y_val_list = y_val.tolist()
        best_f1 = -1.0
        for thr in np.linspace(0.25, 0.75, 51):
            f1 = compute_macro_f1(y_val_list, (val_probs >= thr).astype(int).tolist())
            if f1 > best_f1:
                best_f1 = f1
                self.threshold = float(thr)

    def predict_proba(self, x):
        logits = np.clip(x @ self.w + self.b, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, x):
        return (self.predict_proba(x) >= self.threshold).astype(int)
