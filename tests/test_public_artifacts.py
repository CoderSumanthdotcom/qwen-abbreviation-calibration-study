from __future__ import annotations

import json
import hashlib
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PublicArtifactTests(unittest.TestCase):
    def test_prompt_dataset_shape(self) -> None:
        rows = load_jsonl(ROOT / "data" / "prompts" / "v3_900_prompts.jsonl")
        self.assertEqual(len(rows), 900)
        self.assertEqual(len({row["prompt_id"] for row in rows}), 900)
        self.assertEqual(len({row["set_id"] for row in rows}), 300)
        self.assertEqual(
            Counter(row["condition"] for row in rows),
            {
                "real_low_context": 300,
                "real_high_context": 300,
                "synthetic_high_context": 300,
            },
        )
        self.assertEqual(
            sum("State whether your answer" in row["prompt"] for row in rows),
            0,
        )

    def test_each_public_response_file_has_900_rows(self) -> None:
        response_dir = ROOT / "data" / "responses"
        files = sorted(response_dir.glob("qwen3_*_responses.jsonl"))
        self.assertEqual(len(files), 4)
        for path in files:
            rows = load_jsonl(path)
            self.assertEqual(len(rows), 900, path.name)
            self.assertEqual(len({row["prompt_id"] for row in rows}), 900)
            for row in rows:
                self.assertNotIn("generation_id", row)
                self.assertNotIn("request_id", row)
                self.assertNotIn("run_id", row)

    def test_original_rubric_score_shape(self) -> None:
        rows = load_jsonl(
            ROOT / "data" / "grades" / "original_rubric_scores.jsonl"
        )
        self.assertEqual(len(rows), 3600)
        self.assertEqual(
            Counter(row["condition"] for row in rows),
            {
                "real_low_context": 1200,
                "real_high_context": 1200,
                "synthetic_high_context": 1200,
            },
        )
        ideal_by_condition = Counter(
            row["condition"]
            for row in rows
            if row["accuracy_score_0_2"] == 2
        )
        self.assertEqual(ideal_by_condition["real_low_context"], 473)
        self.assertEqual(ideal_by_condition["real_high_context"], 1118)
        self.assertEqual(ideal_by_condition["synthetic_high_context"], 6)
        self.assertEqual(
            sum(
                row["hallucination_score_0_2"] == 2
                for row in rows
                if row["condition"] == "synthetic_high_context"
            ),
            620,
        )
        self.assertTrue(all("review" not in row for row in rows))

    def test_original_rubric_is_unchanged(self) -> None:
        rubric = (ROOT / "docs" / "grading_rubric.md").read_bytes()
        self.assertEqual(
            hashlib.sha256(rubric).hexdigest(),
            "d458e000316e5879aa837de0e7060348143c2c29654a484657bfbf18a19e272f",
        )

    def test_no_later_validation_workflow_in_project_materials(self) -> None:
        blocked = (
            "adjudicat",
            "human validation",
            "human grading",
            "human-adjudicated",
            "inter-rater",
            "double coding",
            "flagged_for_review",
        )
        paths = [
            ROOT / "README.md",
            ROOT / "data" / "README.md",
            *sorted((ROOT / "docs").glob("*.md")),
            *sorted((ROOT / "poster").glob("*.md")),
            *sorted((ROOT / "scripts").glob("*")),
            *sorted((ROOT / "schemas").glob("*.json")),
        ]
        for path in paths:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8").lower()
            for phrase in blocked:
                self.assertNotIn(phrase, text, f"{phrase!r} found in {path}")

    def test_public_manifests_have_no_local_paths(self) -> None:
        for path in sorted(
            (ROOT / "data" / "responses" / "manifests").glob("*.json")
        ):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/Users/", text)
            self.assertNotIn("generation_id", text)
            self.assertNotIn("request_id", text)

    def test_public_configs_are_dry_run_valid(self) -> None:
        from qwen_abbrev_study.config import load_config
        from qwen_abbrev_study.datasets import load_dataset

        for path in sorted((ROOT / "configs").glob("qwen3_*.json")):
            config = load_config(path)
            self.assertEqual(len(load_dataset(config.dataset)), 900)


if __name__ == "__main__":
    unittest.main()
