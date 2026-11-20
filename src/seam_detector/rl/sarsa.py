# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""On-policy linear SARSA and fixed-heuristic policy evaluation (Proposal Section 4)."""

import numpy as np


class LinearSarsaAgent:
    """On-policy linear SARSA comparator for sequential provenance tagging."""

    def __init__(self, state_dim: int, lr: float = 0.01, discount: float = 1.0):
        self.state_dim = state_dim
        self.lr = lr
        self.discount = discount
        self.weights = np.zeros(2 * state_dim, dtype=np.float64)

    def update(
        self,
        state_vec: np.ndarray,
        action: int,
        reward: float,
        next_state_vec: np.ndarray | None,
        next_action: int | None,
    ) -> float:
        raise NotImplementedError("TODO (Lead 3 — MDP & RL): implement on-policy linear SARSA update.")
