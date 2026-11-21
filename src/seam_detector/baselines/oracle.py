"""Edit-log oracle and oracle-neighbor upper bounds for sequential rollout comparison."""

from seam_detector.features.state_featurizer import build_state_vector


def edit_log_oracle(labels):
    """Return ground-truth labels directly as the provenance upper bound."""
    return list(labels)


def oracle_neighbor_rollout(agent, norm_phi, norm_psi, labels, include_seam=True, include_prev_label=True):
    """Evaluate trained Q-weights when true y*_{t-1} replaces the agent's own rollout guess."""
    preds = []
    n = len(labels)
    for t in range(n):
        prev_lbl = None if t == 0 else labels[t - 1]
        state = build_state_vector(
            norm_phi[t],
            norm_psi[t],
            prev_lbl,
            t / n,
            include_seam=include_seam,
            include_prev_label=include_prev_label,
        )
        preds.append(agent.act(state, epsilon=0.0))
    return preds
