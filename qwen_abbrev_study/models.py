"""Typed records shared by the runner and provider adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


Gateway = Literal["huggingface", "openrouter"]


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    path: Path
    format: Literal["jsonl", "csv"]
    expected_prompt_rows: int | None = None
    expected_set_count: int | None = None


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    temperature: float
    top_p: float
    max_output_tokens: int | None
    thinking_enabled: bool
    reasoning_max_tokens: int | None
    request_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class RetrySettings:
    max_attempts: int
    initial_backoff_seconds: float
    maximum_backoff_seconds: float


@dataclass(frozen=True, slots=True)
class StrategySettings:
    id: str
    prompt_prefix: str = ""
    prompt_suffix: str = ""


@dataclass(frozen=True, slots=True)
class ModelSettings:
    key: str
    gateway: Gateway
    model_id: str
    credential_env: str
    requested_provider: str | None = None
    prompt_prefix: str = ""
    prompt_suffix: str = ""


@dataclass(frozen=True, slots=True)
class StudyConfig:
    schema_version: int
    experiment_id: str
    dataset: DatasetSettings
    output_dir: Path
    randomization_seed: str
    generation: GenerationSettings
    retry: RetrySettings
    strategies: tuple[StrategySettings, ...]
    models: tuple[ModelSettings, ...]
    source_path: Path
    source_sha256: str


@dataclass(frozen=True, slots=True)
class PromptRow:
    prompt_id: str
    set_id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlannedRun:
    request_order: int
    model: ModelSettings
    strategy: StrategySettings
    row: PromptRow
    prompt: str
    prompt_sha256: str
    run_id: str


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    run_id: str
    prompt: str
    model_id: str
    requested_provider_id: str | None
    generation: GenerationSettings
    credential: str


@dataclass(slots=True)
class ProviderResponse:
    completion: str | None
    reasoning: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    finish_reason: str | None
    native_finish_reason: str | None
    actual_model_id: str | None
    actual_provider_id: str | None
    generation_id: str | None
    request_id: str | None
    total_latency_ms: float
    time_to_first_token_ms: float | None


@dataclass(frozen=True, slots=True)
class ErrorDetails:
    type: str
    message: str
    http_status: int | None
    retryable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProviderError(RuntimeError):
    """Normalized provider failure carrying retry and partial-response data."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "provider_error",
        http_status: int | None = None,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
        generation_id: str | None = None,
        request_id: str | None = None,
        actual_model_id: str | None = None,
        actual_provider_id: str | None = None,
        latency_ms: float | None = None,
        time_to_first_token_ms: float | None = None,
        partial_completion: str | None = None,
        partial_reasoning: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.http_status = http_status
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        self.generation_id = generation_id
        self.request_id = request_id
        self.actual_model_id = actual_model_id
        self.actual_provider_id = actual_provider_id
        self.latency_ms = latency_ms
        self.time_to_first_token_ms = time_to_first_token_ms
        self.partial_completion = partial_completion
        self.partial_reasoning = partial_reasoning

    def details(self) -> ErrorDetails:
        return ErrorDetails(
            type=self.error_type,
            message=str(self),
            http_status=self.http_status,
            retryable=self.retryable,
        )
