from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

from qwen_abbrev_study.cli import main
from qwen_abbrev_study.config import ConfigError, load_config
from qwen_abbrev_study.datasets import DatasetError, load_dataset
from qwen_abbrev_study.ids import build_run_id, prompt_sha256
from qwen_abbrev_study.providers.streaming import (
    normalized_usage,
    parse_chat_stream,
    split_thinking_content,
)
from qwen_abbrev_study.runner import build_exact_prompt, plan_runs, run_study
from qwen_abbrev_study.truncation import derive_truncation


def write_fixture(root: Path) -> Path:
    dataset = root / "prompts.jsonl"
    rows = [
        {
            "prompt_id": "PROMPT 1",
            "set_id": "SET_A",
            "prompt": " First prompt\nwith exact whitespace ",
            "condition": "real_low_context",
        },
        {
            "prompt_id": "PROMPT/2",
            "set_id": "SET_B",
            "prompt": "Second prompt",
            "condition": "real_high_context",
        },
    ]
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = {
        "schema_version": 1,
        "experiment_id": "unit-test-experiment-20260723",
        "dataset": {
            "path": "prompts.jsonl",
            "format": "jsonl",
            "expected_prompt_rows": 2,
            "expected_set_count": 2,
        },
        "output_dir": "new-output",
        "randomization_seed": "unit-test-seed",
        "generation": {
            "temperature": 0,
            "top_p": 1,
            "max_output_tokens": 1274,
            "thinking_enabled": True,
            "reasoning_max_tokens": 1024,
            "request_timeout_seconds": 300,
        },
        "retry": {
            "max_attempts": 3,
            "initial_backoff_seconds": 0,
            "maximum_backoff_seconds": 0,
        },
        "strategies": [
            {
                "id": "baseline strategy",
                "prompt_prefix": "[strategy]",
                "prompt_suffix": "[/strategy]",
            }
        ],
        "models": [
            {
                "key": "qwen3_4b",
                "gateway": "huggingface",
                "model_id": "Qwen/Qwen3-4B:featherless-ai",
                "credential_env": "HF_TOKEN",
                "requested_provider": "featherless-ai",
                "prompt_prefix": "[model]",
                "prompt_suffix": "\n\n/think",
            }
        ],
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


class CoreImplementationTests(unittest.TestCase):
    def test_exact_prompt_hash_and_complete_run_id(self) -> None:
        prompt = "Exact prompt\n\n/think"
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self.assertEqual(prompt_sha256(prompt), digest)
        run_id = build_run_id(
            experiment_id="experiment one",
            model_id="Qwen/Qwen3-4B:featherless-ai",
            strategy_id="baseline strategy",
            prompt_id="PROMPT/2",
            prompt_hash=digest,
        )
        self.assertEqual(
            run_id,
            "experiment=experiment%20one"
            "|model=Qwen%2FQwen3-4B%3Afeatherless-ai"
            "|strategy=baseline%20strategy"
            "|prompt=PROMPT%2F2"
            f"|prompt_sha256={digest}",
        )
        self.assertEqual(len(run_id.rsplit("=", 1)[-1]), 64)

    def test_dataset_loader_preserves_exact_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(write_fixture(Path(temporary)))
            rows = load_dataset(config.dataset)
        self.assertEqual(rows[0].prompt, " First prompt\nwith exact whitespace ")
        self.assertEqual(rows[0].metadata["condition"], "real_low_context")

    def test_dataset_loader_rejects_duplicate_prompt_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_fixture(root)
            dataset = root / "prompts.jsonl"
            first = json.loads(dataset.read_text(encoding="utf-8").splitlines()[0])
            dataset.write_text(
                json.dumps(first) + "\n" + json.dumps(first) + "\n",
                encoding="utf-8",
            )
            config = load_config(config_path)
            with self.assertRaisesRegex(DatasetError, "duplicate prompt_id"):
                load_dataset(config.dataset)

    def test_dataset_loader_rejects_whitespace_only_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_fixture(root)
            rows = [
                json.loads(line)
                for line in (root / "prompts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            rows[0]["prompt"] = " \n\t "
            (root / "prompts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            config = load_config(config_path)
            with self.assertRaisesRegex(DatasetError, "prompt must be a nonempty"):
                load_dataset(config.dataset)

    def test_config_rejects_cost_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_fixture(root)
            value = json.loads(config_path.read_text(encoding="utf-8"))
            value["models"][0]["input_cost"] = 1
            config_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "unexpected models"):
                load_config(config_path)

    def test_config_allows_provider_managed_generation_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_fixture(root)
            value = json.loads(config_path.read_text(encoding="utf-8"))
            value["generation"]["max_output_tokens"] = None
            value["generation"]["reasoning_max_tokens"] = None
            config_path.write_text(json.dumps(value), encoding="utf-8")
            config = load_config(config_path)
        self.assertIsNone(config.generation.max_output_tokens)
        self.assertIsNone(config.generation.reasoning_max_tokens)

    def test_plan_hashes_the_final_transmitted_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(write_fixture(Path(temporary)))
            rows = load_dataset(config.dataset)
            plans = plan_runs(config, rows)
        self.assertEqual(len(plans), 2)
        for plan in plans:
            expected = (
                "[model]"
                "[strategy]"
                + plan.row.prompt
                + "[/strategy]"
                + "\n\n/think"
            )
            self.assertEqual(
                build_exact_prompt(plan.row, plan.model, plan.strategy),
                expected,
            )
            self.assertEqual(plan.prompt, expected)
            self.assertEqual(plan.prompt_sha256, prompt_sha256(expected))
            self.assertIn(f"|prompt_sha256={plan.prompt_sha256}", plan.run_id)

    def test_plan_can_select_exact_prompt_ids_without_changing_prompt_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(write_fixture(Path(temporary)))
            rows = load_dataset(config.dataset)
            plans = plan_runs(config, rows, prompt_ids=["PROMPT/2"])
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].row.prompt_id, "PROMPT/2")
        self.assertEqual(
            plans[0].prompt,
            "[model][strategy]Second prompt[/strategy]\n\n/think",
        )

    def test_dry_run_reads_no_credentials_contacts_no_provider_and_writes_nothing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config(write_fixture(root))
            messages: list[str] = []

            def forbidden_provider(_: str) -> object:
                self.fail("dry run attempted to create a provider")

            with patch.dict(os.environ, {}, clear=True):
                summary = run_study(
                    config,
                    provider_factory=forbidden_provider,  # type: ignore[arg-type]
                    emit=messages.append,
                )
            self.assertFalse(summary.execute)
            self.assertEqual(summary.planned_runs, 2)
            self.assertFalse(config.output_dir.exists())
            self.assertIn("no credentials were read", messages[-1])

    def test_cli_defaults_to_safe_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_fixture(root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch.dict(os.environ, {}, clear=True):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = main(["--config", str(config_path)])
            self.assertEqual(exit_code, 0)
            self.assertIn("DRY RUN", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")
            self.assertFalse((root / "new-output").exists())

    def test_execute_without_credentials_stops_before_provider_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_fixture(root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch.dict(os.environ, {}, clear=True):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = main(["--config", str(config_path), "--execute"])
            self.assertEqual(exit_code, 2)
            self.assertIn("HF_TOKEN is required", stderr.getvalue())
            self.assertFalse((root / "new-output").exists())

    def test_stream_parser_collects_reasoning_completion_usage_and_ttft(self) -> None:
        events = [
            'data: {"id":"gen-1","model":"actual/model","provider":"provider-x",'
            '"choices":[{"delta":{"reasoning":"step "}}]}',
            "",
            'data: {"choices":[{"delta":{"reasoning":"two"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"answer"},'
            '"finish_reason":"length","native_finish_reason":"max_tokens"}],'
            '"usage":{"prompt_tokens":10,"completion_tokens":7,'
            '"total_tokens":17,"completion_tokens_details":{"reasoning_tokens":2}}}',
            "",
            "data: [DONE]",
            "",
        ]
        parsed = parse_chat_stream(
            events,
            request_started=1.0,
            clock=lambda: 1.25,
        )
        self.assertEqual(parsed.reasoning, "step two")
        self.assertEqual(parsed.completion, "answer")
        self.assertEqual(parsed.actual_model_id, "actual/model")
        self.assertEqual(parsed.actual_provider_id, "provider-x")
        self.assertEqual(parsed.completion_id, "gen-1")
        self.assertEqual(parsed.time_to_first_token_ms, 250)
        self.assertEqual(
            normalized_usage(parsed.usage),
            (10, 7, 2, 17),
        )
        self.assertTrue(
            derive_truncation(parsed.finish_reason, parsed.native_finish_reason)
        )

    def test_think_tags_are_separated_without_answer_length_metrics(self) -> None:
        completion, reasoning = split_thinking_content(
            "<think>internal reasoning</think>\nFinal answer",
            None,
        )
        self.assertEqual(completion, "Final answer")
        self.assertEqual(reasoning, "internal reasoning")

    def test_truncation_is_three_valued_and_conservative(self) -> None:
        cases = [
            (("length", None), True),
            (("stop", None), False),
            ((None, "MAX_TOKENS"), True),
            (("unknown-provider-value", None), None),
            ((None, None), None),
            (("stop", "max_tokens"), True),
        ]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                self.assertIs(derive_truncation(*arguments), expected)


if __name__ == "__main__":
    unittest.main()
