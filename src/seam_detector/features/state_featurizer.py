"""Assemble the 32-dim MDP state vector combining line cues and seam history."""

import numpy as np
from seam_detector.features.line_features import FEATURE_NAMES
from seam_detector.features.seam_features import SEAM_FEATURE_NAMES
from seam_detector.schema import Label


def state_feature_names():
    names = list(FEATURE_NAMES)
    names.extend([f"seam_{n}" for n in SEAM_FEATURE_NAMES])
    names.extend([f"prev_x_{n}" for n in SEAM_FEATURE_NAMES])
    names.extend(["prev_pred_signed", "is_first_line", "rel_position", "bias"])
    return names


def build_state_vector(
    norm_phi_t,
    norm_psi_t,
    prev_label,
    rel_pos,
    include_seam=True,
    include_prev_label=True,
):
    psi_part = norm_psi_t if include_seam else np.zeros_like(norm_psi_t)
    if prev_label is None or not include_prev_label:
        prev_signed = 0.0
        is_first = 1.0 if prev_label is None else 0.0
    else:
        prev_signed = 1.0 if prev_label == Label.AI else -1.0
        is_first = 0.0

    interaction = prev_signed * psi_part if (include_seam and include_prev_label) else np.zeros_like(psi_part)
    tail = np.array([prev_signed, is_first, rel_pos, 1.0], dtype=np.float64)
    return np.concatenate([norm_phi_t, psi_part, interaction, tail])
