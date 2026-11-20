# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""Sequential line-labeling MDP environment (Proposal Table 4)."""

from seam_detector.schema import Label, Passage


class SeamDetectionMDP:
    """Episode over passage s = (c_1, ..., c_n) with deterministic left-to-right transitions.

    Rewards:
      - Step reward r_t = +1 if a_t == y*_t else -1
      - Terminal bonus r_{n+1} = MacroF1(y_hat_{1..n}, y*_{1..n})
    """

    def __init__(self, passage: Passage, discount: float = 1.0):
        self.passage = passage
        self.discount = discount

    def reset(self):
        raise NotImplementedError("TODO (Lead 3 — MDP & RL): initialize episode state s_1.")

    def step(self, action: Label):
        raise NotImplementedError("TODO (Lead 3 — MDP & RL): advance to s_{t+1} and compute reward.")
