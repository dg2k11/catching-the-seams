"""On-policy linear SARSA agent reusing shared linear action-value storage."""

from seam_detector.rl.q_learning import LinearRLAgent


class LinearSarsaAgent(LinearRLAgent):
    def update_sarsa(self, state_vec, action, reward, next_state_vec, next_action, step_lr=None):
        lr = self.lr if step_lr is None else step_lr
        if next_state_vec is None or next_action is None:
            target = reward
        else:
            target = reward + self.discount * float(self.w[next_action] @ next_state_vec)
        return self._td_update(state_vec, action, target, lr)
