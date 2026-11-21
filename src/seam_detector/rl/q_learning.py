"""Linear Q-learning agent with shared linear TD update helper."""

import numpy as np


class LinearRLAgent:
    def __init__(self, state_dim, lr=0.006, discount=0.9, l2=1e-4, seed=221):
        self.state_dim = state_dim
        self.lr = lr
        self.discount = discount
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros((2, state_dim), dtype=np.float64)

    def q_values(self, state_vec):
        return self.w @ state_vec

    def act(self, state_vec, epsilon=0.0):
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(0, 2))
        q = self.q_values(state_vec)
        return 1 if q[1] >= q[0] else 0

    def _td_update(self, state_vec, action, target, step_lr):
        q_sa = float(self.w[action] @ state_vec)
        td_err = float(np.clip(q_sa - target, -5.0, 5.0))
        self.w[action] -= step_lr * (td_err * state_vec + self.l2 * self.w[action])
        return td_err

    def update(self, state_vec, action, reward, next_state_vec, step_lr=None):
        lr = self.lr if step_lr is None else step_lr
        if next_state_vec is None:
            target = reward
        else:
            target = reward + self.discount * float(np.max(self.q_values(next_state_vec)))
        return self._td_update(state_vec, action, target, lr)
