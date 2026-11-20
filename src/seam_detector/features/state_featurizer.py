# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""State vector f(s_t) and action-block template Phi(s_t, a) (Proposal Appendix A.2)."""

import numpy as np
from seam_detector.schema import Label


def build_state_vector(
    phi_t: np.ndarray,
    psi_t: np.ndarray,
    prev_label: int,
    relative_position: float,
    include_seam: bool = True,
    include_prev_label: bool = True,
) -> np.ndarray:
    """Assemble f(s_t) with bias term and optional E2 ablation masks."""
    raise NotImplementedError("TODO (Lead 2/3 — Features & MDP): implement f(s_t) assembly and ablation flags.")


def build_action_features(state_vec: np.ndarray, action: Label) -> np.ndarray:
    """Place a copy of f(s_t) into the action block for action a in {Human, AI}."""
    raise NotImplementedError("TODO (Lead 2/3 — Features & MDP): implement CS221 action-block template Phi(s_t, a).")
