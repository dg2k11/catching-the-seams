"""Data containers for passages, line labels, and explanations."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class Label(IntEnum):
    HUMAN = 0
    AI = 1


@dataclass
class LineRecord:
    line_index: int
    text: str
    label: Label
    source_model: str = "Human_story"
    was_rewritten: bool = False


@dataclass
class Passage:
    passage_id: str
    slice_name: str
    edit_rate: int
    split: str
    lines: list[LineRecord] = field(default_factory=list)
    is_gold_human_edit: bool = False

    @property
    def num_lines(self):
        return len(self.lines)

    @property
    def true_ai_share(self):
        if not self.lines:
            return 0.0
        return sum(1 for ln in self.lines if ln.label == Label.AI) / len(self.lines)


@dataclass
class LinePrediction:
    line_index: int
    predicted_label: Label
    margin_delta_q: float
    top_contributions: list[tuple[str, float]] = field(default_factory=list)
    ground_truth: Optional[Label] = None
