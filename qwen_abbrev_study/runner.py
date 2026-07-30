"""Study planning, retries, resume behavior, and normalized result records."""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .datasets import load_dataset
from .ids import build_run_id, prompt_sha256
from .models import (
    ModelSettings,
    PlannedRun,
    PromptRow,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    StrategySettings,
    StudyConfig,
)
from .providers import Provider, create_provider
from .storage import ExperimentStorage, sha256_file, utc_now
from .truncation import derive_truncation


ProviderFactory = Callable[[str], Provider]


class StudyExecutionError(RuntimeError):
    """Raised for safe preflight or execution failures."""


@dataclass(frozen=True, slots=True)
class RunSummary:
    execute: bool
    prompt_rows: int
    set_count: int
    selected_models: tuple[str, ...]
    selected_strategies: tuple[str, ...]
    planned_runs: int
    skipped_successes: int = 0
    skipped_blocked: int = 0
    completed_successes: int = 0
    completed_failures: int = 0


def build_exact_prompt(
    row: PromptRow,
    model: ModelSettings,
    strategy: StrategySettings,
) -> str:
    """Apply configured fragments exactly, without implicit separators."""

    return (
        model.prompt_prefix
        + strategy.prompt_prefix
        + row.prompt
        + strategy.prompt_suffix
        + model.prompt_suffix
    )


def _stable_shuffle(rows: Sequence[PromptRow], seed_text: str) -> list[PromptRow]:
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest(), "big")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _select(
    items: Sequence[ModelSettings] | Sequence[StrategySettings],
    selected: Sequence[str] | None,
    *,
    attribute: str,
    kind: str,
) -> list[Any]:
    if selected is None:
        return list(items)
    lookup = {getattr(item, attribute): item for item in items}
    missing = [key for key in selected if key not in lookup]
    if missing:
        raise StudyExecutionError(f"unknown {kind} value(s): {', '.join(missing)}")
    return [lookup[key] for key in selected]


def plan_runs(
    config: StudyConfig,
    rows: Sequence[PromptRow],
    *,
    model_keys: Sequence[str] | None = None,
    strategy_ids: Sequence[str] | None = None,
    prompt_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[PlannedRun]:
    """Create deterministic, independently randomized model/strategy plans."""

    if limit is not None and limit < 1:
        raise StudyExecutionError("limit must be a positive integer")
    models = _select(
        config.models,
        model_keys,
        attribute="key",
        kind="model key",
    )
    strategies = _select(
        config.strategies,
        strategy_ids,
        attribute="id",
        kind="strategy id",
    )
    selected_prompt_ids: set[str] | None = None
    if prompt_ids is not None:
        if not prompt_ids:
            raise StudyExecutionError("prompt ID selection must not be empty")
        selected_prompt_ids = set(prompt_ids)
        if len(selected_prompt_ids) != len(prompt_ids):
            raise StudyExecutionError("prompt ID selection contains duplicates")
        known_prompt_ids = {row.prompt_id for row in rows}
        missing_prompt_ids = sorted(selected_prompt_ids - known_prompt_ids)
        if missing_prompt_ids:
            raise StudyExecutionError(
                "unknown prompt ID value(s): " + ", ".join(missing_prompt_ids)
            )
    plans: list[PlannedRun] = []
    for model in models:
        for strategy in strategies:
            seed = (
                f"{config.randomization_seed}|{model.model_id}|{strategy.id}"
            )
            ordered_rows = _stable_shuffle(rows, seed)
            if selected_prompt_ids is not None:
                ordered_rows = [
                    row for row in ordered_rows if row.prompt_id in selected_prompt_ids
                ]
            if limit is not None:
                ordered_rows = ordered_rows[:limit]
            for request_order, row in enumerate(ordered_rows, start=1):
                prompt = build_exact_prompt(row, model, strategy)
                prompt_hash = prompt_sha256(prompt)
                plans.append(
                    PlannedRun(
                        request_order=request_order,
                        model=model,
                        strategy=strategy,
                        row=row,
                        prompt=prompt,
                        prompt_sha256=prompt_hash,
                        run_id=build_run_id(
                            experiment_id=config.experiment_id,
                            model_id=model.model_id,
                            strategy_id=strategy.id,
                            prompt_id=row.prompt_id,
                            prompt_hash=prompt_hash,
                        ),
                    )
                )
    return plans


def _attempt_base(plan: PlannedRun, attempt_number: int, started_at: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": plan.run_id,
        "experiment_id": None,
        "model_key": plan.model.key,
        "strategy_id": plan.strategy.id,
        "prompt_id": plan.row.prompt_id,
        "prompt_sha256": plan.prompt_sha256,
        "attempt_number": attempt_number,
        "success": False,
        "status": "failure",
        "generation_id": None,
        "request_id": None,
        "actual_model_id": None,
        "actual_provider_id": None,
        "finish_reason": None,
        "native_finish_reason": None,
        "truncated": None,
        "latency_ms": 0.0,
        "time_to_first_token_ms": None,
        "error": None,
        "started_at_utc": started_at,
        "completed_at_utc": started_at,
    }


def _success_attempt(
    config: StudyConfig,
    plan: PlannedRun,
    attempt_number: int,
    started_at: str,
    completed_at: str,
    response: ProviderResponse,
) -> dict[str, Any]:
    result = _attempt_base(plan, attempt_number, started_at)
    result.update(
        {
            "experiment_id": config.experiment_id,
            "success": True,
            "status": "success",
            "generation_id": response.generation_id,
            "request_id": response.request_id,
            "actual_model_id": response.actual_model_id,
            "actual_provider_id": response.actual_provider_id,
            "finish_reason": response.finish_reason,
            "native_finish_reason": response.native_finish_reason,
            "truncated": derive_truncation(
                response.finish_reason,
                response.native_finish_reason,
            ),
            "latency_ms": response.total_latency_ms,
            "time_to_first_token_ms": response.time_to_first_token_ms,
            "completed_at_utc": completed_at,
        }
    )
    return result


def _failure_attempt(
    config: StudyConfig,
    plan: PlannedRun,
    attempt_number: int,
    started_at: str,
    completed_at: str,
    error: ProviderError,
    fallback_latency_ms: float,
) -> dict[str, Any]:
    result = _attempt_base(plan, attempt_number, started_at)
    result.update(
        {
            "experiment_id": config.experiment_id,
            "generation_id": error.generation_id,
            "request_id": error.request_id,
            "actual_model_id": error.actual_model_id,
            "actual_provider_id": error.actual_provider_id,
            "latency_ms": (
                error.latency_ms
                if error.latency_ms is not None
                else fallback_latency_ms
            ),
            "time_to_first_token_ms": error.time_to_first_token_ms,
            "error": error.details().to_dict(),
            "completed_at_utc": completed_at,
        }
    )
    return result


def _terminal_result(
    *,
    config: StudyConfig,
    plan: PlannedRun,
    success: bool,
    attempts: int,
    failed_attempts: int,
    response: ProviderResponse | None,
    error: ProviderError | None,
    operation_started_at: str,
    operation_completed_at: str,
    operation_elapsed_ms: float,
    terminal_latency_ms: float | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": plan.run_id,
        "experiment_id": config.experiment_id,
        "model_key": plan.model.key,
        "strategy_id": plan.strategy.id,
        "prompt_id": plan.row.prompt_id,
        "set_id": plan.row.set_id,
        "request_order": plan.request_order,
        "dataset_prompt": plan.row.prompt,
        "prompt": plan.prompt,
        "prompt_sha256": plan.prompt_sha256,
        "dataset_metadata": plan.row.metadata,
        "gateway": plan.model.gateway,
        "requested_model_id": plan.model.model_id,
        "actual_model_id": (
            response.actual_model_id
            if response is not None
            else error.actual_model_id if error is not None else None
        ),
        "requested_provider_id": plan.model.requested_provider,
        "actual_provider_id": (
            response.actual_provider_id
            if response is not None
            else error.actual_provider_id if error is not None else None
        ),
        "generation_id": (
            response.generation_id
            if response is not None
            else error.generation_id if error is not None else None
        ),
        "request_id": (
            response.request_id
            if response is not None
            else error.request_id if error is not None else None
        ),
        "success": success,
        "status": "success" if success else "failure",
        "attempts": attempts,
        "retries": max(0, attempts - 1),
        "failed_attempts": failed_attempts,
        "error": None if success else error.details().to_dict() if error else None,
        "completion": (
            response.completion
            if response is not None
            else error.partial_completion if error is not None else None
        ),
        "reasoning": (
            response.reasoning
            if response is not None
            else error.partial_reasoning if error is not None else None
        ),
        "prompt_tokens": response.prompt_tokens if response is not None else None,
        "completion_tokens": response.completion_tokens if response is not None else None,
        "reasoning_tokens": response.reasoning_tokens if response is not None else None,
        "total_tokens": response.total_tokens if response is not None else None,
        "finish_reason": response.finish_reason if response is not None else None,
        "native_finish_reason": (
            response.native_finish_reason if response is not None else None
        ),
        "truncated": (
            derive_truncation(response.finish_reason, response.native_finish_reason)
            if response is not None
            else None
        ),
        "total_latency_ms": terminal_latency_ms,
        "time_to_first_token_ms": (
            response.time_to_first_token_ms
            if response is not None
            else error.time_to_first_token_ms if error is not None else None
        ),
        "operation_elapsed_ms": operation_elapsed_ms,
        "started_at_utc": operation_started_at,
        "completed_at_utc": operation_completed_at,
    }


def run_one(
    config: StudyConfig,
    plan: PlannedRun,
    *,
    provider: Provider,
    credential: str,
    storage: ExperimentStorage,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run one prompt with append-only attempt auditing and bounded retries."""

    operation_started = clock()
    operation_started_at = utc_now()
    failed_attempts = 0
    response: ProviderResponse | None = None
    terminal_error: ProviderError | None = None
    terminal_latency_ms: float | None = None
    attempts = 0

    for attempt_number in range(1, config.retry.max_attempts + 1):
        attempts = attempt_number
        attempt_started = clock()
        attempt_started_at = utc_now()
        try:
            response = provider.complete(
                ProviderRequest(
                    run_id=plan.run_id,
                    prompt=plan.prompt,
                    model_id=plan.model.model_id,
                    requested_provider_id=plan.model.requested_provider,
                    generation=config.generation,
                    credential=credential,
                )
            )
            terminal_latency_ms = response.total_latency_ms
            attempt_completed_at = utc_now()
            storage.append_attempt(
                _success_attempt(
                    config,
                    plan,
                    attempt_number,
                    attempt_started_at,
                    attempt_completed_at,
                    response,
                )
            )
            terminal_error = None
            break
        except ProviderError as error:
            failed_attempts += 1
            terminal_error = error
            terminal_latency_ms = (
                error.latency_ms
                if error.latency_ms is not None
                else max(0.0, (clock() - attempt_started) * 1000)
            )
            attempt_completed_at = utc_now()
            storage.append_attempt(
                _failure_attempt(
                    config,
                    plan,
                    attempt_number,
                    attempt_started_at,
                    attempt_completed_at,
                    error,
                    terminal_latency_ms,
                )
            )
            if not error.retryable or attempt_number >= config.retry.max_attempts:
                break
            backoff = min(
                config.retry.maximum_backoff_seconds,
                config.retry.initial_backoff_seconds * (2 ** (attempt_number - 1)),
            )
            if error.retry_after_seconds is not None:
                backoff = min(
                    config.retry.maximum_backoff_seconds,
                    error.retry_after_seconds,
                )
            sleep(backoff)

    success = response is not None
    operation_completed_at = utc_now()
    result = _terminal_result(
        config=config,
        plan=plan,
        success=success,
        attempts=attempts,
        failed_attempts=failed_attempts,
        response=response,
        error=terminal_error,
        operation_started_at=operation_started_at,
        operation_completed_at=operation_completed_at,
        operation_elapsed_ms=max(0.0, (clock() - operation_started) * 1000),
        terminal_latency_ms=terminal_latency_ms,
    )
    storage.append_result(result)
    return result


_SYSTEMIC_HTTP_STATUSES = {400, 401, 402, 403, 404, 422, 429}
_SYSTEMIC_ERROR_PATTERN = re.compile(
    r"(?:authentication|unauthorized|forbidden|invalid api key|"
    r"insufficient (?:credit|fund)|no endpoints? found|rate.?limit)",
    re.IGNORECASE,
)
_CREDIT_ERROR_PATTERN = re.compile(
    r"(?:insufficient (?:credit|credits|fund|funds)|"
    r"(?:credit|credits|quota) (?:exhausted|depleted|exceeded)|"
    r"out of credits|payment required)",
    re.IGNORECASE,
)


def _blocks_remaining_model_requests(result: dict[str, Any]) -> bool:
    """Identify failures likely to repeat for every prompt on the same route."""

    error = result.get("error")
    if not isinstance(error, dict):
        return False
    status = error.get("http_status")
    message = error.get("message")
    return (
        status in _SYSTEMIC_HTTP_STATUSES
        or isinstance(message, str)
        and _SYSTEMIC_ERROR_PATTERN.search(message) is not None
    )


def _is_credit_exhaustion(result: dict[str, Any]) -> bool:
    """Stop the entire experiment when a provider requires more paid credit."""

    error = result.get("error")
    if not isinstance(error, dict):
        return False
    status = error.get("http_status")
    message = error.get("message")
    return status == 402 or (
        isinstance(message, str)
        and _CREDIT_ERROR_PATTERN.search(message) is not None
    )


def run_study(
    config: StudyConfig,
    *,
    execute: bool = False,
    resume: bool = False,
    rerun: bool = False,
    model_keys: Sequence[str] | None = None,
    strategy_ids: Sequence[str] | None = None,
    prompt_ids: Sequence[str] | None = None,
    limit: int | None = None,
    provider_factory: ProviderFactory = create_provider,
    emit: Callable[[str], None] = print,
) -> RunSummary:
    """Validate by default; contact providers only when execute=True."""

    rows = load_dataset(config.dataset)
    plans = plan_runs(
        config,
        rows,
        model_keys=model_keys,
        strategy_ids=strategy_ids,
        prompt_ids=prompt_ids,
        limit=limit,
    )
    selected_models = tuple(dict.fromkeys(plan.model.key for plan in plans))
    selected_strategies = tuple(dict.fromkeys(plan.strategy.id for plan in plans))
    base_summary = RunSummary(
        execute=execute,
        prompt_rows=len(rows),
        set_count=len({row.set_id for row in rows}),
        selected_models=selected_models,
        selected_strategies=selected_strategies,
        planned_runs=len(plans),
    )
    emit(
        f"Validated {base_summary.prompt_rows} prompt rows across "
        f"{base_summary.set_count} sets; planned {base_summary.planned_runs} requests."
    )
    dataset_hash = sha256_file(config.dataset.path)
    emit(f"Dataset SHA-256: {dataset_hash}")
    if not execute:
        emit("DRY RUN: no credentials were read and no provider was contacted.")
        return base_summary
    if resume is False and rerun:
        raise StudyExecutionError("--rerun requires --resume")

    credentials: dict[str, str] = {}
    for plan in plans:
        env_name = plan.model.credential_env
        if env_name not in credentials:
            value = os.environ.get(env_name)
            if not value:
                raise StudyExecutionError(
                    f"{env_name} is required only because --execute was supplied"
                )
            credentials[env_name] = value

    storage = ExperimentStorage(config)
    manifest = storage.initialize(
        resume=resume,
        dataset_sha256=dataset_hash,
        selected_model_keys=list(selected_models),
        selected_strategy_ids=list(selected_strategies),
        planned_run_count=len(plans),
    )
    latest = storage.latest_results()
    successful_ids = {
        run_id
        for run_id, result in latest.items()
        if result.get("success") is True
    }
    providers: dict[str, Provider] = {}
    skipped = 0
    skipped_blocked = 0
    successes = 0
    failures = 0
    blocked_models: set[str] = set()
    try:
        for index, plan in enumerate(plans, start=1):
            if not rerun and plan.run_id in successful_ids:
                skipped += 1
                emit(f"[{index}/{len(plans)}] skip successful {plan.row.prompt_id}")
                continue
            if plan.model.key in blocked_models:
                skipped_blocked += 1
                emit(
                    f"[{index}/{len(plans)}] skip blocked model "
                    f"{plan.model.key}/{plan.row.prompt_id}"
                )
                continue
            provider = providers.get(plan.model.gateway)
            if provider is None:
                provider = provider_factory(plan.model.gateway)
                providers[plan.model.gateway] = provider
            emit(
                f"[{index}/{len(plans)}] {plan.model.key}/"
                f"{plan.strategy.id}/{plan.row.prompt_id}"
            )
            result = run_one(
                config,
                plan,
                provider=provider,
                credential=credentials[plan.model.credential_env],
                storage=storage,
            )
            if result["success"]:
                successes += 1
            else:
                failures += 1
                if _is_credit_exhaustion(result):
                    emit(
                        "PAUSED: provider credits or paid quota are exhausted. "
                        "No later requests will be started; add credit only if "
                        "the human operator chooses to, then use --execute --resume."
                    )
                    raise StudyExecutionError(
                        "provider credits or paid quota exhausted; execution paused"
                    )
                if _blocks_remaining_model_requests(result):
                    blocked_models.add(plan.model.key)
                    emit(
                        f"Stopping remaining requests for {plan.model.key}; "
                        "the terminal provider error is likely route-wide. "
                        "Fix the issue and use --execute --resume."
                    )
    finally:
        for provider in providers.values():
            provider.close()

    latest = storage.latest_results()
    manifest.update(
        {
            "completed_at_utc": utc_now(),
            "success_count": sum(result.get("success") is True for result in latest.values()),
            "failure_count": sum(result.get("success") is False for result in latest.values()),
        }
    )
    storage.write_manifest(manifest)
    emit(
        f"Execution finished: {successes} succeeded, {failures} failed, "
        f"{skipped} previously successful runs skipped, "
        f"{skipped_blocked} blocked-model runs deferred."
    )
    return RunSummary(
        execute=True,
        prompt_rows=base_summary.prompt_rows,
        set_count=base_summary.set_count,
        selected_models=selected_models,
        selected_strategies=selected_strategies,
        planned_runs=base_summary.planned_runs,
        skipped_successes=skipped,
        skipped_blocked=skipped_blocked,
        completed_successes=successes,
        completed_failures=failures,
    )
