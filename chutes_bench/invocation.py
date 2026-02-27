"""Data structures for capturing LLM invocation details."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMInvocation:
    """Snapshot of a single LLM API call — request, response, and metadata."""

    request_messages: list[dict] = field(default_factory=list)
    response_raw: dict = field(default_factory=dict)
    model_api_id: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
