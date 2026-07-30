"""Crash-resilient manifest and append-only JSONL persistence."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import StudyConfig


class StorageError(RuntimeError):
    """Raised when experiment isolation or persisted data is unsafe."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise StorageError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise StorageError(f"invalid JSON in {path}: {error.msg}") from error
    if not isinstance(value, dict):
        raise StorageError(f"{path} must contain a JSON object")
    return value


class ExperimentStorage:
    """Own one isolated experiment output directory."""

    def __init__(self, config: StudyConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.manifest_path = self.output_dir / "manifest.json"
        self.results_path = self.output_dir / "results.jsonl"
        self.attempts_path = self.output_dir / "attempts.jsonl"

    def initialize(
        self,
        *,
        resume: bool,
        dataset_sha256: str,
        selected_model_keys: list[str],
        selected_strategy_ids: list[str],
        planned_run_count: int,
    ) -> dict[str, Any]:
        if resume:
            if not self.output_dir.is_dir() or not self.manifest_path.is_file():
                raise StorageError(
                    f"--resume requires an existing experiment manifest: {self.manifest_path}"
                )
            manifest = _read_json(self.manifest_path)
            expected = {
                "experiment_id": self.config.experiment_id,
                "config_sha256": self.config.source_sha256,
                "dataset_sha256": dataset_sha256,
            }
            mismatches = {
                key: (manifest.get(key), value)
                for key, value in expected.items()
                if manifest.get(key) != value
            }
            if mismatches:
                raise StorageError(
                    "resume manifest does not match the current run: "
                    + json.dumps(mismatches, sort_keys=True)
                )
            return manifest

        if self.output_dir.exists():
            raise StorageError(
                f"fresh execution requires a new output directory; already exists: "
                f"{self.output_dir}"
            )
        self.output_dir.mkdir(parents=True)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "experiment_id": self.config.experiment_id,
            "execution_mode": "execute",
            "config_path": str(self.config.source_path),
            "config_sha256": self.config.source_sha256,
            "dataset_path": str(self.config.dataset.path),
            "dataset_sha256": dataset_sha256,
            "selected_model_keys": selected_model_keys,
            "selected_strategy_ids": selected_strategy_ids,
            "randomization_seed": self.config.randomization_seed,
            "planned_run_count": planned_run_count,
            "created_at_utc": utc_now(),
            "completed_at_utc": None,
            "success_count": 0,
            "failure_count": 0,
        }
        self.write_manifest(manifest)
        return manifest

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        temporary = self.manifest_path.with_name(f".{self.manifest_path.name}.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    def append_result(self, result: dict[str, Any]) -> None:
        self._append_jsonl(self.results_path, result)

    def append_attempt(self, attempt: dict[str, Any]) -> None:
        self._append_jsonl(self.attempts_path, attempt)

    @staticmethod
    def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def latest_results(self) -> dict[str, dict[str, Any]]:
        if not self.results_path.exists():
            return {}
        latest: dict[str, dict[str, Any]] = {}
        with self.results_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise StorageError(
                        f"{self.results_path}:{line_number}: invalid JSON: {error.msg}"
                    ) from error
                if not isinstance(value, dict) or not isinstance(value.get("run_id"), str):
                    raise StorageError(
                        f"{self.results_path}:{line_number}: result must contain run_id"
                    )
                latest[value["run_id"]] = value
        return latest
