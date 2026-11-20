# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""Shared data containers for passages, per-line ground truth, and explainable predictions."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class Label(IntEnum):
    """Line-level provenance action/label space: {Human (0), AI (1)}."""

    HUMAN = 0
    AI = 1


@dataclass
class LineRecord:
    """Single sentence c_t within a passage s = (c_1, ..., c_n)."""

    line_index: int
    text: str
    label: Label
    source_model: str = "Human_story"
    was_rewritten: bool = False


@dataclass
class Passage:
    """benchmark episode representing a 6-20 sentence passage."""

    passage_id: str
    slice_name: str               # "H", "E_0", "E_25", "E_50", "E_75", or "case_study"
    edit_rate: int                # k in {0, 25, 50, 75}
    split: str                    # "train", "validation", or "test"
    lines: list[LineRecord] = field(default_factory=list)
    is_gold_human_edit: bool = False

    @property
    def num_lines(self) -> int:
        return len(self.lines)

    @property
    def true_ai_share(self) -> float:
        if not self.lines:
            return 0.0
        return sum(1 for line in self.lines if line.label == Label.AI) / len(self.lines)


@dataclass
class LinePrediction:
    """Agent prediction y_hat_t and exact linear feature attributions e_t."""

    line_index: int
    predicted_label: Label
    margin_delta_q: float
    top_contributions: list[tuple[str, float]] = field(default_factory=list)
    ground_truth: Optional[Label] = None
