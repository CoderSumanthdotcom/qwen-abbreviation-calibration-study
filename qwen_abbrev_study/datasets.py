"""Load and validate prepared prompt datasets without changing prompt text."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .models import DatasetSettings, PromptRow


class DatasetError(ValueError):
    """Raised when a prepared dataset violates the input contract."""


def _nonempty_string(value: Any, *, field: str, row_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"row {row_number}: {field} must be a nonempty string")
    return value


def _rows_from_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise DatasetError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise DatasetError(f"{path}:{line_number}: each JSONL row must be an object")
            yield line_number, value


def _rows_from_csv(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"{path}: CSV header is missing")
        for row_number, value in enumerate(reader, start=2):
            yield row_number, dict(value)


def load_dataset(settings: DatasetSettings) -> list[PromptRow]:
    """Load JSONL or CSV rows and preserve each exact prompt string."""

    path = settings.path
    if not path.is_file():
        raise DatasetError(f"dataset does not exist: {path}")
    source = _rows_from_jsonl(path) if settings.format == "jsonl" else _rows_from_csv(path)
    rows: list[PromptRow] = []
    seen_prompt_ids: set[str] = set()
    for row_number, value in source:
        prompt_id = _nonempty_string(value.get("prompt_id"), field="prompt_id", row_number=row_number)
        set_id = _nonempty_string(value.get("set_id"), field="set_id", row_number=row_number)
        prompt = _nonempty_string(value.get("prompt"), field="prompt", row_number=row_number)
        if prompt_id in seen_prompt_ids:
            raise DatasetError(f"row {row_number}: duplicate prompt_id {prompt_id!r}")
        seen_prompt_ids.add(prompt_id)
        metadata = {
            key: item
            for key, item in value.items()
            if key not in {"prompt_id", "set_id", "prompt"}
        }
        rows.append(
            PromptRow(
                prompt_id=prompt_id,
                set_id=set_id,
                prompt=prompt,
                metadata=metadata,
            )
        )
    if not rows:
        raise DatasetError(f"dataset contains no prompt rows: {path}")
    if settings.expected_prompt_rows is not None and len(rows) != settings.expected_prompt_rows:
        raise DatasetError(
            f"expected {settings.expected_prompt_rows} prompt rows, found {len(rows)}"
        )
    set_count = len({row.set_id for row in rows})
    if settings.expected_set_count is not None and set_count != settings.expected_set_count:
        raise DatasetError(
            f"expected {settings.expected_set_count} sets, found {set_count}"
        )
    return rows
