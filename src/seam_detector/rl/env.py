"""Sequential line-labeling MDP environment with step rewards and terminal Macro-F1 bonus."""

from seam_detector.eval.metrics import compute_macro_f1
from seam_detector.features.state_featurizer import build_state_vector


class SeamEpisodeEnv:
    def __init__(
        self,
        norm_phi,
        norm_psi,
        labels,
        terminal_bonus_scale=2.0,
        include_seam=True,
        include_prev_label=True,
    ):
        self.norm_phi = norm_phi
        self.norm_psi = norm_psi
        self.labels = labels
        self.n = len(labels)
        self.terminal_bonus_scale = terminal_bonus_scale
        self.include_seam = include_seam
        self.include_prev_label = include_prev_label
        self.t = 0
        self.preds = []

    def reset(self):
        self.t = 0
        self.preds = []
        return build_state_vector(
            self.norm_phi[0],
            self.norm_psi[0],
            None,
            0.0,
            include_seam=self.include_seam,
            include_prev_label=self.include_prev_label,
        )

    def step(self, action):
        y_true = self.labels[self.t]
        reward = 1.0 if action == y_true else -1.0
        self.preds.append(action)
        self.t += 1
        done = self.t >= self.n
        if done:
            f1_bonus = compute_macro_f1(self.labels, self.preds)
            reward += self.terminal_bonus_scale * (f1_bonus - 0.5)
            return None, reward, True
        next_state = build_state_vector(
            self.norm_phi[self.t],
            self.norm_psi[self.t],
            action,
            self.t / self.n,
            include_seam=self.include_seam,
            include_prev_label=self.include_prev_label,
        )
        return next_state, reward, False
