"""Derive truncation from normalized and provider-native finish reasons."""

from __future__ import annotations

import re


_TRUNCATED = {
    "length",
    "max_tokens",
    "max_token",
    "max_output_tokens",
    "token_limit",
    "output_limit",
    "model_length",
}

_NORMAL = {
    "stop",
    "stopped",
    "eos",
    "end_turn",
    "end",
    "complete",
    "completed",
    "tool_calls",
    "tool_call",
}


def _normalize(reason: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", reason.strip().lower()).strip("_")


def derive_truncation(
    finish_reason: str | None,
    native_finish_reason: str | None,
) -> bool | None:
    """Return True, False, or None when provider reasons are inconclusive."""

    reasons = [
        _normalize(value)
        for value in (finish_reason, native_finish_reason)
        if isinstance(value, str) and value.strip()
    ]
    if not reasons:
        return None
    if any(reason in _TRUNCATED for reason in reasons):
        return True
    if any(reason in _NORMAL for reason in reasons):
        return False
    return None

