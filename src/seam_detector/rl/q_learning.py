# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""Epsilon-greedy Q-learning with linear function approximation (Proposal Section 4, Eq. 1)."""

import numpy as np


class LinearQLearningAgent:
    """Linear Q-agent Q_w(s, a) = w^T Phi(s, a) trained on CPU in NumPy."""

    def __init__(self, state_dim: int, lr: float = 0.01, discount: float = 1.0):
        self.state_dim = state_dim
        self.lr = lr
        self.discount = discount
        self.weights = np.zeros(2 * state_dim, dtype=np.float64)

    def act(self, state_vec: np.ndarray, epsilon: float = 0.0) -> int:
        raise NotImplementedError("TODO (Lead 3 — MDP & RL): implement epsilon-greedy action selection.")

    def update(self, state_vec: np.ndarray, action: int, reward: float, next_state_vec: np.ndarray | None) -> float:
        """Apply w <- w - eta * [Q_w(s, a) - (r + gamma * max_{a'} Q_w(s', a'))] * Phi(s, a)."""
        raise NotImplementedError("TODO (Lead 3 — MDP & RL): implement linear Q-learning TD update.")
