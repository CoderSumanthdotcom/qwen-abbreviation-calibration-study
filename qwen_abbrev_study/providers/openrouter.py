"""OpenRouter adapter for Qwen3 8B, 14B, and 32B routes."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from ..models import ProviderError, ProviderRequest, ProviderResponse
from .http import elapsed_ms, response_error, transport_error
from .streaming import normalized_usage, parse_chat_stream


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_GENERATION_URL = "https://openrouter.ai/api/v1/generation"


class OpenRouterProvider:
    """Stream chat completions and normalize OpenRouter generation metadata."""

    def __init__(self, *, client: httpx.Client | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.Client()

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        started = time.monotonic()
        reasoning: dict[str, Any]
        if request.generation.thinking_enabled:
            reasoning = {
                "enabled": True,
                "exclude": False,
            }
            if request.generation.reasoning_max_tokens is not None:
                reasoning["max_tokens"] = request.generation.reasoning_max_tokens
        else:
            reasoning = {"effort": "none", "exclude": False}
        payload = {
            "model": request.model_id,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.generation.temperature,
            "top_p": request.generation.top_p,
            "reasoning": reasoning,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.requested_provider_id:
            payload["provider"] = {
                "order": [request.requested_provider_id],
                "only": [request.requested_provider_id],
                "allow_fallbacks": False,
            }
        if request.generation.max_output_tokens is not None:
            payload["max_tokens"] = request.generation.max_output_tokens
        headers = {
            "Authorization": f"Bearer {request.credential}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get(
                "OPENROUTER_SITE_URL",
                "https://localhost/qwen-abbreviation-study",
            ),
            "X-Title": "Qwen3 Abbreviation Study",
        }
        try:
            with self._client.stream(
                "POST",
                OPENROUTER_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=request.generation.request_timeout_seconds,
            ) as response:
                generation_id = response.headers.get("x-generation-id")
                request_id = response.headers.get("x-request-id")
                if not response.is_success:
                    raise response_error(
                        response,
                        gateway="OpenRouter",
                        started=started,
                        generation_id=generation_id,
                        request_id=request_id,
                    )
                try:
                    streamed = parse_chat_stream(
                        response.iter_lines(),
                        request_started=started,
                    )
                except ProviderError as error:
                    error.generation_id = error.generation_id or generation_id
                    error.request_id = error.request_id or request_id
                    error.latency_ms = error.latency_ms or elapsed_ms(started)
                    raise
        except ProviderError:
            raise
        except httpx.HTTPError as error:
            raise transport_error(error, gateway="OpenRouter", started=started) from error

        response_latency_ms = elapsed_ms(started)
        generation_id = generation_id or streamed.completion_id
        stats = self._generation_stats(
            credential=request.credential,
            generation_id=generation_id,
            timeout=request.generation.request_timeout_seconds,
        )
        prompt_tokens, completion_tokens, reasoning_tokens, total_tokens = normalized_usage(
            streamed.usage,
            stats,
        )
        stats_model = stats.get("model") if stats else None
        stats_provider = stats.get("provider_name") if stats else None
        stats_native_finish = stats.get("native_finish_reason") if stats else None
        return ProviderResponse(
            completion=streamed.completion,
            reasoning=streamed.reasoning,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            finish_reason=streamed.finish_reason,
            native_finish_reason=(
                stats_native_finish
                if isinstance(stats_native_finish, str) and stats_native_finish
                else streamed.native_finish_reason
            ),
            actual_model_id=(
                stats_model
                if isinstance(stats_model, str) and stats_model
                else streamed.actual_model_id
            ),
            actual_provider_id=(
                stats_provider
                if isinstance(stats_provider, str) and stats_provider
                else streamed.actual_provider_id
            ),
            generation_id=generation_id,
            request_id=request_id,
            total_latency_ms=response_latency_ms,
            time_to_first_token_ms=streamed.time_to_first_token_ms,
        )

    def _generation_stats(
        self,
        *,
        credential: str,
        generation_id: str | None,
        timeout: float,
    ) -> dict[str, Any] | None:
        if not generation_id:
            return None
        try:
            response = self._client.get(
                OPENROUTER_GENERATION_URL,
                params={"id": generation_id},
                headers={"Authorization": f"Bearer {credential}"},
                timeout=timeout,
            )
            if not response.is_success:
                return None
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        if not isinstance(body, dict):
            return None
        data = body.get("data", body)
        return data if isinstance(data, dict) else None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
