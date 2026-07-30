"""Configuration loading with structural and semantic validation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, cast

from .models import (
    DatasetSettings,
    GenerationSettings,
    ModelSettings,
    RetrySettings,
    StrategySettings,
    StudyConfig,
)


class ConfigError(ValueError):
    """Raised when configuration cannot safely define a study run."""


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a nonempty array")
    return value


def _string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a nonempty string"
        raise ConfigError(f"{field} must be {qualifier}")
    return value


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: Any, field: str, *, minimum: float, exclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number")
    numeric = float(value)
    if (exclusive and numeric <= minimum) or (not exclusive and numeric < minimum):
        operator = ">" if exclusive else ">="
        raise ConfigError(f"{field} must be {operator} {minimum}")
    return numeric


def _optional_positive_integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field, minimum=1)


def _ensure_unique(values: list[str], field: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ConfigError(f"{field} values must be unique; duplicates: {duplicates}")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _reject_unexpected(value: dict[str, Any], allowed: set[str], field: str) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ConfigError(f"unexpected {field} fields: {unexpected}")


def load_config(path: str | Path) -> StudyConfig:
    """Load a versioned config while keeping secret values out of memory."""

    source_path = Path(path).expanduser().resolve()
    try:
        raw_bytes = source_path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read config {source_path}: {error}") from error
    try:
        root = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid JSON in {source_path}: {error.msg}") from error
    root = _object(root, "config")
    allowed_root = {
        "schema_version",
        "experiment_id",
        "dataset",
        "output_dir",
        "randomization_seed",
        "generation",
        "retry",
        "strategies",
        "models",
    }
    unexpected = sorted(set(root) - allowed_root)
    if unexpected:
        raise ConfigError(f"unexpected config fields: {unexpected}")
    if root.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    experiment_id = _string(root.get("experiment_id"), "experiment_id")
    randomization_seed = _string(root.get("randomization_seed"), "randomization_seed")
    base = source_path.parent

    dataset_raw = _object(root.get("dataset"), "dataset")
    _reject_unexpected(
        dataset_raw,
        {"path", "format", "expected_prompt_rows", "expected_set_count"},
        "dataset",
    )
    dataset_path_text = _string(dataset_raw.get("path"), "dataset.path")
    format_value = dataset_raw.get("format")
    if format_value is None:
        suffix = Path(dataset_path_text).suffix.lower()
        format_value = "jsonl" if suffix in {".jsonl", ".ndjson"} else "csv" if suffix == ".csv" else None
    if format_value not in {"jsonl", "csv"}:
        raise ConfigError("dataset.format must be jsonl or csv")
    dataset = DatasetSettings(
        path=_resolve(base, dataset_path_text),
        format=cast(Literal["jsonl", "csv"], format_value),
        expected_prompt_rows=_optional_positive_integer(
            dataset_raw.get("expected_prompt_rows"), "dataset.expected_prompt_rows"
        ),
        expected_set_count=_optional_positive_integer(
            dataset_raw.get("expected_set_count"), "dataset.expected_set_count"
        ),
    )

    generation_raw = _object(root.get("generation"), "generation")
    _reject_unexpected(
        generation_raw,
        {
            "temperature",
            "top_p",
            "max_output_tokens",
            "thinking_enabled",
            "reasoning_max_tokens",
            "request_timeout_seconds",
        },
        "generation",
    )
    top_p = _number(generation_raw.get("top_p"), "generation.top_p", minimum=0, exclusive=True)
    if top_p > 1:
        raise ConfigError("generation.top_p must be <= 1")
    thinking_enabled = generation_raw.get("thinking_enabled")
    if not isinstance(thinking_enabled, bool):
        raise ConfigError("generation.thinking_enabled must be a boolean")
    generation = GenerationSettings(
        temperature=_number(
            generation_raw.get("temperature"), "generation.temperature", minimum=0
        ),
        top_p=top_p,
        max_output_tokens=_optional_positive_integer(
            generation_raw.get("max_output_tokens"),
            "generation.max_output_tokens",
        ),
        thinking_enabled=thinking_enabled,
        reasoning_max_tokens=_optional_positive_integer(
            generation_raw.get("reasoning_max_tokens"),
            "generation.reasoning_max_tokens",
        ),
        request_timeout_seconds=_number(
            generation_raw.get("request_timeout_seconds"),
            "generation.request_timeout_seconds",
            minimum=0,
            exclusive=True,
        ),
    )
    if (
        generation.thinking_enabled
        and generation.reasoning_max_tokens is not None
        and generation.max_output_tokens is not None
        and generation.reasoning_max_tokens >= generation.max_output_tokens
    ):
        raise ConfigError(
            "generation.reasoning_max_tokens must be less than max_output_tokens "
            "when thinking is enabled"
        )

    retry_raw = _object(root.get("retry"), "retry")
    _reject_unexpected(
        retry_raw,
        {
            "max_attempts",
            "initial_backoff_seconds",
            "maximum_backoff_seconds",
        },
        "retry",
    )
    retry = RetrySettings(
        max_attempts=_integer(retry_raw.get("max_attempts"), "retry.max_attempts", minimum=1),
        initial_backoff_seconds=_number(
            retry_raw.get("initial_backoff_seconds"),
            "retry.initial_backoff_seconds",
            minimum=0,
        ),
        maximum_backoff_seconds=_number(
            retry_raw.get("maximum_backoff_seconds"),
            "retry.maximum_backoff_seconds",
            minimum=0,
        ),
    )
    if retry.maximum_backoff_seconds < retry.initial_backoff_seconds:
        raise ConfigError(
            "retry.maximum_backoff_seconds must be >= initial_backoff_seconds"
        )

    strategies: list[StrategySettings] = []
    for index, value in enumerate(_list(root.get("strategies"), "strategies")):
        item = _object(value, f"strategies[{index}]")
        _reject_unexpected(
            item,
            {"id", "prompt_prefix", "prompt_suffix"},
            f"strategies[{index}]",
        )
        strategies.append(
            StrategySettings(
                id=_string(item.get("id"), f"strategies[{index}].id"),
                prompt_prefix=_string(
                    item.get("prompt_prefix", ""),
                    f"strategies[{index}].prompt_prefix",
                    allow_empty=True,
                ),
                prompt_suffix=_string(
                    item.get("prompt_suffix", ""),
                    f"strategies[{index}].prompt_suffix",
                    allow_empty=True,
                ),
            )
        )
    _ensure_unique([strategy.id for strategy in strategies], "strategy id")

    models: list[ModelSettings] = []
    credential_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for index, value in enumerate(_list(root.get("models"), "models")):
        item = _object(value, f"models[{index}]")
        _reject_unexpected(
            item,
            {
                "key",
                "gateway",
                "model_id",
                "credential_env",
                "requested_provider",
                "prompt_prefix",
                "prompt_suffix",
            },
            f"models[{index}]",
        )
        gateway = item.get("gateway")
        if gateway not in {"huggingface", "openrouter"}:
            raise ConfigError(f"models[{index}].gateway must be huggingface or openrouter")
        credential_env = _string(item.get("credential_env"), f"models[{index}].credential_env")
        if not credential_pattern.fullmatch(credential_env):
            raise ConfigError(f"models[{index}].credential_env is not a valid environment name")
        requested_provider = item.get("requested_provider")
        if requested_provider is not None:
            requested_provider = _string(
                requested_provider, f"models[{index}].requested_provider"
            )
        models.append(
            ModelSettings(
                key=_string(item.get("key"), f"models[{index}].key"),
                gateway=cast(Literal["huggingface", "openrouter"], gateway),
                model_id=_string(item.get("model_id"), f"models[{index}].model_id"),
                credential_env=credential_env,
                requested_provider=requested_provider,
                prompt_prefix=_string(
                    item.get("prompt_prefix", ""),
                    f"models[{index}].prompt_prefix",
                    allow_empty=True,
                ),
                prompt_suffix=_string(
                    item.get("prompt_suffix", ""),
                    f"models[{index}].prompt_suffix",
                    allow_empty=True,
                ),
            )
        )
    _ensure_unique([model.key for model in models], "model key")
    _ensure_unique([model.model_id for model in models], "model id")

    output_dir = _resolve(base, _string(root.get("output_dir"), "output_dir"))
    if output_dir == dataset.path or output_dir == source_path:
        raise ConfigError("output_dir must not overwrite the dataset or configuration file")

    return StudyConfig(
        schema_version=1,
        experiment_id=experiment_id,
        dataset=dataset,
        output_dir=output_dir,
        randomization_seed=randomization_seed,
        generation=generation,
        retry=retry,
        strategies=tuple(strategies),
        models=tuple(models),
        source_path=source_path,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
