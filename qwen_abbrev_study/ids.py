"""Stable prompt hashes and unambiguous run identifiers."""

from __future__ import annotations

import hashlib
from urllib.parse import quote


def prompt_sha256(prompt: str) -> str:
    """Return the full SHA-256 digest of the exact UTF-8 prompt."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_run_id(
    *,
    experiment_id: str,
    model_id: str,
    strategy_id: str,
    prompt_id: str,
    prompt_hash: str,
) -> str:
    """Build the complete, non-truncated run ID required by the study."""

    if len(prompt_hash) != 64 or any(char not in "0123456789abcdef" for char in prompt_hash):
        raise ValueError("prompt_hash must be a 64-character lowercase SHA-256 digest")
    encode = lambda value: quote(value, safe="")
    return (
        f"experiment={encode(experiment_id)}"
        f"|model={encode(model_id)}"
        f"|strategy={encode(strategy_id)}"
        f"|prompt={encode(prompt_id)}"
        f"|prompt_sha256={prompt_hash}"
    )

