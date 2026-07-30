"""OpenAI-compatible SSE parsing and response normalization helpers."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..models import ProviderError


@dataclass(slots=True)
class StreamedChat:
    completion: str
    reasoning: str | None
    usage: dict[str, Any] | None
    finish_reason: str | None
    native_finish_reason: str | None
    actual_model_id: str | None
    actual_provider_id: str | None
    completion_id: str | None
    time_to_first_token_ms: float | None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_text(item) for item in value)
    if isinstance(value, dict):
        return _text(
            value.get("text")
            or value.get("content")
            or value.get("reasoning")
            or value.get("data")
        )
    return ""


def _iter_sse_payloads(lines: Iterable[str]) -> Iterable[str]:
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


def parse_chat_stream(
    lines: Iterable[str],
    *,
    request_started: float,
    clock: Any = time.monotonic,
) -> StreamedChat:
    """Parse an OpenAI-compatible SSE stream without discarding raw text."""

    completion_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    actual_model_id: str | None = None
    actual_provider_id: str | None = None
    completion_id: str | None = None
    first_token_ms: float | None = None

    for payload in _iter_sse_payloads(lines):
        if not payload or payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProviderError(
                f"invalid JSON in provider stream: {error.msg}",
                error_type="invalid_stream",
                retryable=True,
                partial_completion="".join(completion_parts) or None,
                partial_reasoning="".join(reasoning_parts) or None,
                generation_id=completion_id,
                actual_model_id=actual_model_id,
                actual_provider_id=actual_provider_id,
                time_to_first_token_ms=first_token_ms,
            ) from error
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            error_value = chunk["error"]
            if isinstance(error_value, dict):
                message = str(error_value.get("message") or error_value)
                status_value = error_value.get("code")
                status = status_value if isinstance(status_value, int) else None
            else:
                message = str(error_value)
                status = None
            raise ProviderError(
                message,
                error_type="stream_error",
                http_status=status,
                retryable=status is None or status in {408, 409, 425, 429, 500, 502, 503, 504},
                partial_completion="".join(completion_parts) or None,
                partial_reasoning="".join(reasoning_parts) or None,
                generation_id=completion_id,
                actual_model_id=actual_model_id,
                actual_provider_id=actual_provider_id,
                time_to_first_token_ms=first_token_ms,
            )

        model_value = chunk.get("model")
        provider_value = chunk.get("provider")
        id_value = chunk.get("id")
        if actual_model_id is None and isinstance(model_value, str) and model_value:
            actual_model_id = model_value
        if actual_provider_id is None and isinstance(provider_value, str) and provider_value:
            actual_provider_id = provider_value
        if completion_id is None and isinstance(id_value, str) and id_value:
            completion_id = id_value
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        completion_piece = _text(delta.get("content"))
        reasoning_piece = _text(
            delta.get("reasoning")
            or delta.get("reasoning_content")
            or delta.get("reasoning_details")
        )
        if completion_piece:
            completion_parts.append(completion_piece)
        if reasoning_piece:
            reasoning_parts.append(reasoning_piece)
        if first_token_ms is None and (completion_piece or reasoning_piece):
            first_token_ms = max(0.0, (clock() - request_started) * 1000)

        choice_finish = choice.get("finish_reason")
        choice_native_finish = choice.get("native_finish_reason")
        if isinstance(choice_finish, str) and choice_finish:
            finish_reason = choice_finish
        if isinstance(choice_native_finish, str) and choice_native_finish:
            native_finish_reason = choice_native_finish

    explicit_reasoning = "".join(reasoning_parts) or None
    completion, reasoning = split_thinking_content(
        "".join(completion_parts),
        explicit_reasoning,
    )
    return StreamedChat(
        completion=completion,
        reasoning=reasoning,
        usage=usage,
        finish_reason=finish_reason,
        native_finish_reason=native_finish_reason,
        actual_model_id=actual_model_id,
        actual_provider_id=actual_provider_id,
        completion_id=completion_id,
        time_to_first_token_ms=first_token_ms,
    )


_CLOSED_THINK = re.compile(r"^\s*<think>([\s\S]*?)</think>\s*", re.IGNORECASE)
_OPEN_THINK = re.compile(r"^\s*<think>([\s\S]*)$", re.IGNORECASE)


def split_thinking_content(
    completion: str,
    explicit_reasoning: str | None,
) -> tuple[str, str | None]:
    """Move a leading Qwen think block into the normalized reasoning field."""

    match = _CLOSED_THINK.match(completion)
    tagged_reasoning: str | None = None
    if match:
        tagged_reasoning = match.group(1)
        completion = completion[match.end() :]
    else:
        open_match = _OPEN_THINK.match(completion)
        if open_match:
            tagged_reasoning = open_match.group(1)
            completion = ""
    parts = [
        value
        for value in (explicit_reasoning, tagged_reasoning)
        if isinstance(value, str) and value != ""
    ]
    reasoning = "\n".join(parts) if parts else None
    return completion, reasoning


def nullable_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def normalized_usage(
    usage: dict[str, Any] | None,
    provider_stats: dict[str, Any] | None = None,
) -> tuple[int | None, int | None, int | None, int | None]:
    """Normalize prompt/completion/reasoning/total tokens."""

    usage = usage or {}
    provider_stats = provider_stats or {}
    details = (
        usage.get("completion_tokens_details")
        if isinstance(usage.get("completion_tokens_details"), dict)
        else {}
    )
    prompt_tokens = nullable_int(
        provider_stats.get("native_tokens_prompt", usage.get("prompt_tokens"))
    )
    completion_tokens = nullable_int(
        provider_stats.get("native_tokens_completion", usage.get("completion_tokens"))
    )
    reasoning_tokens = nullable_int(
        provider_stats.get(
            "native_tokens_reasoning",
            details.get("reasoning_tokens", details.get("reasoning")),
        )
    )
    total_tokens = nullable_int(usage.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, reasoning_tokens, total_tokens
