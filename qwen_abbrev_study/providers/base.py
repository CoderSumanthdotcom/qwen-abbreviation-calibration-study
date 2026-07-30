"""Provider interface and adapter factory."""

from __future__ import annotations

from typing import Protocol

import httpx

from ..models import Gateway, ProviderRequest, ProviderResponse


class Provider(Protocol):
    """Synchronous provider interface used by the sequential study runner."""

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Send one fresh-context request and return normalized measurements."""

    def close(self) -> None:
        """Release HTTP resources."""


def create_provider(
    gateway: Gateway,
    *,
    client: httpx.Client | None = None,
) -> Provider:
    """Create the adapter for a configured gateway."""

    if gateway == "huggingface":
        from .huggingface import HuggingFaceProvider

        return HuggingFaceProvider(client=client)
    if gateway == "openrouter":
        from .openrouter import OpenRouterProvider

        return OpenRouterProvider(client=client)
    raise ValueError(f"unsupported gateway: {gateway}")

