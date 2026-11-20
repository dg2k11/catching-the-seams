# Non-core scaffolding stub (AI-generated boilerplate; core implementation to be written by the student team per the CS221 Classroom Track policy).
"""Smoke tests verifying package layout, config schema, and data containers."""

import pathlib
import unittest
import yaml

from seam_detector.schema import Label, LineRecord, Passage


class RepoSmokeTest(unittest.TestCase):
    def test_passage_ai_share(self):
        lines = [
            LineRecord(0, "Red dust clung to his ankles.", Label.HUMAN),
            LineRecord(1, "It was not the heat that bowed him.", Label.AI),
            LineRecord(2, "He spat into the weeds. Nothing moved.", Label.HUMAN),
            LineRecord(3, "In that fleeting moment, the grove whispered.", Label.AI),
        ]
        passage = Passage(
            passage_id="demo_001",
            slice_name="E_50",
            edit_rate=50,
            split="validation",
            lines=lines,
        )
        self.assertEqual(passage.num_lines, 4)
        self.assertAlmostEqual(passage.true_ai_share, 0.50)

    def test_default_config_loads(self):
        cfg_path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self.assertEqual(cfg["dataset"]["hf_repo"], "Rajarshi-Roy-research/Defactify_Text_Dataset")
        self.assertEqual(cfg["dataset"]["edit_rates"], [0, 25, 50, 75])
        self.assertEqual(cfg["mdp"]["actions"], ["Human", "AI"])


if __name__ == "__main__":
    unittest.main()
