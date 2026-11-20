# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""Cross-sentence seam shift features psi(c_{t-1}, c_t) (5 dimensions; Proposal Table 2 & Appendix A.2)."""

import numpy as np


def extract_seam_features(sentences: list[str], line_features: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Compute psi(c_{t-1}, c_t) capturing human<->AI style jumps across adjacent sentences.

    Planned seam features:
      1. |Delta curvature|
      2. |Delta sentence length|
      3. Cosine distance between adjacent sentence embeddings
      4. |Delta dependency-tree depth|
      5. Previous-line sentence-fragment indicator
    """
    raise NotImplementedError("TODO (Lead 2 — Features): implement seam shift features psi(c_{t-1}, c_t).")
