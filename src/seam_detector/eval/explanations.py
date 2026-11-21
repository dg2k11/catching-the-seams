"""Additive per-feature decision decomposition and E3 faithfulness flip check."""

import random
import numpy as np
from seam_detector.schema import Label, LinePrediction


def explain_line_decision(
    state_vec,
    w_human,
    w_ai,
    feature_names,
    line_index,
    ground_truth=None,
    top_k=4,
):
    contribs = (w_ai - w_human) * state_vec
    margin = float(np.sum(contribs))
    pred = Label.AI if margin >= 0.0 else Label.HUMAN
    ranked = [(feature_names[k], float(contribs[k])) for k in range(len(feature_names) - 1)]
    ranked.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return LinePrediction(
        line_index=line_index,
        predicted_label=pred,
        margin_delta_q=margin,
        top_contributions=ranked[:top_k],
        ground_truth=Label(ground_truth) if ground_truth is not None else None,
    )


def evaluate_faithfulness_flip_rate(states, w_human, w_ai, seed=221):
    rng = random.Random(seed)
    diff_w = w_ai - w_human
    d_non_bias = len(diff_w) - 1
    top_flips = 0
    rand_flips = 0
    total = len(states)

    for s in states:
        contribs = diff_w * s
        orig_margin = float(np.sum(contribs))
        orig_label = int(orig_margin >= 0.0)
        top_idx = int(np.argmax(contribs[:d_non_bias]) if orig_label == 1 else np.argmin(contribs[:d_non_bias]))
        if int((orig_margin - float(contribs[top_idx])) >= 0.0) != orig_label:
            top_flips += 1
        rand_idx = rng.randrange(d_non_bias)
        if int((orig_margin - float(contribs[rand_idx])) >= 0.0) != orig_label:
            rand_flips += 1

    return {
        "top_feature_flip_rate": top_flips / total if total else 0.0,
        "random_feature_flip_rate": rand_flips / total if total else 0.0,
        "num_lines": total,
    }
