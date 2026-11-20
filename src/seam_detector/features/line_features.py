# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""Per-line feature vector phi(c_t) (~20 dimensions; Proposal Appendix A.2)."""

import numpy as np


def extract_line_features(sentences: list[str]) -> tuple[np.ndarray, list[str]]:
    """Compute phi(c_t) for each sentence c_t in a passage.

    Planned feature groups (Appendix A.2):
      - Offline frozen Gemma 3 1B Fast-DetectGPT curvature and mean token log-probability
      - Sentence token length and deviation from passage mean length
      - Type-token ratio, concrete-to-abstract noun ratio, dependency-tree depth
      - Binary rhetorical indicators (antithesis, tricolon, participial tail,
        passive voice, sentence-initial adverbial opener)
    """
    raise NotImplementedError("TODO (Lead 2 — Features): implement per-line feature extraction phi(c_t).")
