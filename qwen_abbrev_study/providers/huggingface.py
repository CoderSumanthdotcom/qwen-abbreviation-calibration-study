"""Hugging Face router adapter pinned to Featherless by the model route."""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..models import ProviderError, ProviderRequest, ProviderResponse
from .http import elapsed_ms, response_error, transport_error
from .streaming import normalized_usage, parse_chat_stream


HUGGINGFACE_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


class HuggingFaceProvider:
    """Stream chat completions through Hugging Face inference providers."""

    def __init__(self, *, client: httpx.Client | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.Client()

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        started = time.monotonic()
        payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.generation.temperature,
            "top_p": request.generation.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if not request.generation.thinking_enabled:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if request.generation.max_output_tokens is not None:
            payload["max_tokens"] = request.generation.max_output_tokens
        try:
            with self._client.stream(
                "POST",
                HUGGINGFACE_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {request.credential}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=request.generation.request_timeout_seconds,
            ) as response:
                request_id = (
                    response.headers.get("x-request-id")
                    or response.headers.get("x-inference-id")
                )
                generation_id = response.headers.get("x-generation-id")
                if not response.is_success:
                    raise response_error(
                        response,
                        gateway="Hugging Face",
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
            raise transport_error(error, gateway="Hugging Face", started=started) from error

        prompt_tokens, completion_tokens, reasoning_tokens, total_tokens = normalized_usage(
            streamed.usage
        )
        return ProviderResponse(
            completion=streamed.completion,
            reasoning=streamed.reasoning,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            finish_reason=streamed.finish_reason,
            native_finish_reason=streamed.native_finish_reason,
            actual_model_id=streamed.actual_model_id,
            actual_provider_id=streamed.actual_provider_id
            or request.requested_provider_id,
            generation_id=generation_id or streamed.completion_id,
            request_id=request_id,
            total_latency_ms=elapsed_ms(started),
            time_to_first_token_ms=streamed.time_to_first_token_ms,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
