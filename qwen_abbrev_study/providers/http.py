"""Shared HTTP error handling for provider adapters."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from ..models import ProviderError


RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


def elapsed_ms(started: float) -> float:
    return max(0.0, (time.monotonic() - started) * 1000)


def retry_after_seconds(headers: httpx.Headers) -> float | None:
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def response_error(
    response: httpx.Response,
    *,
    gateway: str,
    started: float,
    generation_id: str | None,
    request_id: str | None,
) -> ProviderError:
    try:
        body_text = response.read().decode("utf-8", errors="replace")
    except httpx.HTTPError:
        body_text = ""
    message = body_text[:2000]
    try:
        body: Any = json.loads(body_text)
    except json.JSONDecodeError:
        body = None
    if isinstance(body, dict):
        error_value = body.get("error")
        if isinstance(error_value, dict):
            message = str(error_value.get("message") or error_value)
        elif error_value:
            message = str(error_value)
        elif body.get("message"):
            message = str(body["message"])
    message = message or response.reason_phrase or "request failed"
    return ProviderError(
        f"{gateway} HTTP {response.status_code}: {message}",
        error_type="http_error",
        http_status=response.status_code,
        retryable=response.status_code in RETRYABLE_HTTP_STATUSES,
        retry_after_seconds=retry_after_seconds(response.headers),
        generation_id=generation_id,
        request_id=request_id,
        latency_ms=elapsed_ms(started),
    )


def transport_error(error: Exception, *, gateway: str, started: float) -> ProviderError:
    if isinstance(error, httpx.TimeoutException):
        error_type = "timeout"
    else:
        error_type = "transport_error"
    return ProviderError(
        f"{gateway} {error_type}: {error}",
        error_type=error_type,
        retryable=True,
        latency_ms=elapsed_ms(started),
    )

