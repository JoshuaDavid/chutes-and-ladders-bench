"""LLM player implementations — OpenAI, Anthropic, and OpenRouter."""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from chutes_bench.board import CHUTES_LADDERS
from chutes_bench.invocation import LLMInvocation
from chutes_bench.tools import TOOL_SCHEMAS


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert SDK objects (with model_dump) to plain dicts/lists."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj

# ── System prompt ────────────────────────────────────────────────────

_CHUTES_TABLE = "\n".join(f"  {sq} → {dest}" for sq, dest in sorted(CHUTES_LADDERS.items()))

SYSTEM_PROMPT = f"""\
You are playing Chutes & Ladders on a 10×10 board (squares 1–100).

Rules:
- Spinner gives 1–6.
- Your first spin puts you on that square number.
- You must land EXACTLY on 100 to win. If the spin would take you past 100, you stay put.
- No extra turn for spinning 6.
- If you land on a chute/ladder base, you MUST take it.
- An illegal move is an automatic loss.

Chutes & Ladders map:
{_CHUTES_TABLE}

Turn sequence:
1. Call spin_spinner to get your spin value.
2. Call move_pawn_to_square with (your current square + spin value).
   - If the spin overshoots 100, pass your current square (you stay put).
3. If you landed on a ladder base, call ascend_ladder_to_square with the destination.
   If you landed on a chute top, call descend_chute_to_square with the destination.
4. Call end_turn.

You may also call send_message, forfeit, offer_draw, or accept_draw at any time.
You may call plan at any time to think step-by-step before acting. It has no side effects.
Play to win. Do not forfeit or offer draws unless the situation is truly hopeless.
"""


# ── OpenAI-compatible player (works for OpenAI + OpenRouter) ─────────

@dataclass
class OpenAIPlayer:
    """Player backed by any OpenAI-compatible chat/completions API."""

    model: str
    display_name: str
    api_key: str | None = None
    base_url: str | None = None
    _messages: list[dict] = field(default_factory=list, repr=False)
    _client: object = field(default=None, repr=False)
    _last_invocation: LLMInvocation | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.display_name

    @property
    def last_invocation(self) -> LLMInvocation | None:
        return self._last_invocation

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def _init_messages(self):
        if not self._messages:
            self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def next_action(self, observation: str) -> tuple[str, dict]:
        self._init_messages()
        # If the last message is an assistant with tool_calls that never got
        # a tool result (e.g. turn ended without observe()), inject one now.
        if self._messages and self._messages[-1].get("role") == "assistant":
            tc = self._messages[-1].get("tool_calls", [])
            if tc:
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc[0]["id"],
                    "content": "OK",
                })
        self._messages.append({"role": "user", "content": observation})

        # Snapshot messages before the API call
        request_snapshot = _to_json_safe(self._messages)

        client = self._get_client()
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            tools=TOOL_SCHEMAS,
            tool_choice="required",
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        msg = response.choices[0].message

        # Capture invocation metadata
        usage = getattr(response, "usage", None)
        response_dict = response.model_dump() if hasattr(response, "model_dump") else {}
        self._last_invocation = LLMInvocation(
            request_messages=request_snapshot,
            response_raw=response_dict,
            model_api_id=self.model,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=latency_ms,
        )

        # Append assistant message to history, keeping only the first
        # tool call so every tool_call_id gets a corresponding response.
        msg_dict = msg.model_dump(exclude_none=True)
        if msg_dict.get("tool_calls") and len(msg_dict["tool_calls"]) > 1:
            msg_dict["tool_calls"] = msg_dict["tool_calls"][:1]
        self._messages.append(msg_dict)

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            return tool_name, args

        # No tool call — treat as forfeit (shouldn't happen with tool_choice=required)
        return "forfeit", {}

    def observe(self, message: str) -> None:
        self._init_messages()
        # Add tool result to conversation
        if self._messages and self._messages[-1].get("role") == "assistant":
            last = self._messages[-1]
            tool_calls = last.get("tool_calls", [])
            if tool_calls:
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tool_calls[0]["id"],
                    "content": message,
                })
                return
        # Fallback: add as user message
        self._messages.append({"role": "user", "content": message})

    def reset(self) -> None:
        self._messages = []
        self._last_invocation = None


# ── Anthropic player ─────────────────────────────────────────────────

def _openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI tool schemas to Anthropic format."""
    result = []
    for t in tools:
        fn = t["function"]
        result.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        })
    return result


@dataclass
class AnthropicPlayer:
    """Player backed by Anthropic's messages API."""

    model: str
    display_name: str
    api_key: str | None = None
    _messages: list[dict] = field(default_factory=list, repr=False)
    _client: object = field(default=None, repr=False)
    _last_invocation: LLMInvocation | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.display_name

    @property
    def last_invocation(self) -> LLMInvocation | None:
        return self._last_invocation

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def next_action(self, observation: str) -> tuple[str, dict]:
        # If the last message is an assistant with a tool_use that never got
        # a tool_result (e.g. turn ended without observe()), inject one
        # combined with the new observation to keep alternating roles.
        if self._messages and self._messages[-1].get("role") == "assistant":
            content = self._messages[-1].get("content", [])
            for block in content:
                if getattr(block, "type", None) == "tool_use":
                    self._messages.append({
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": block.id, "content": "OK"},
                            {"type": "text", "text": observation},
                        ],
                    })
                    break
            else:
                self._messages.append({"role": "user", "content": observation})
        else:
            self._messages.append({"role": "user", "content": observation})

        # Snapshot messages before the API call
        request_snapshot = _to_json_safe(self._messages)

        client = self._get_client()
        t0 = time.monotonic()
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=self._messages,
            tools=_openai_tools_to_anthropic(TOOL_SCHEMAS),
            tool_choice={"type": "any"},
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Capture invocation metadata
        usage = getattr(response, "usage", None)
        response_dict = response.model_dump() if hasattr(response, "model_dump") else {}
        self._last_invocation = LLMInvocation(
            request_messages=request_snapshot,
            response_raw=response_dict,
            model_api_id=self.model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            latency_ms=latency_ms,
        )

        # Build assistant message for history, keeping only the first
        # tool_use block so every tool_use_id gets a tool_result.
        first_tool = None
        kept_content = []
        for block in response.content:
            if block.type == "tool_use":
                if first_tool is None:
                    first_tool = block
                    kept_content.append(block)
                # skip additional tool_use blocks
            else:
                kept_content.append(block)
        self._messages.append({"role": "assistant", "content": kept_content})

        if first_tool is not None:
            self._last_tool_use_id = first_tool.id
            return first_tool.name, first_tool.input
        return "forfeit", {}

    def observe(self, message: str) -> None:
        self._messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": self._last_tool_use_id, "content": message}],
        })

    def reset(self) -> None:
        self._messages = []
        self._last_invocation = None


# ── Model registry ───────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """How to instantiate a player for a given model."""

    id: str
    display_name: str
    provider: str  # "openai" | "anthropic" | "openrouter"

    def make_player(self) -> OpenAIPlayer | AnthropicPlayer:
        if self.provider == "anthropic":
            return AnthropicPlayer(
                model=self.id,
                display_name=self.display_name,
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
        elif self.provider == "openrouter":
            return OpenAIPlayer(
                model=self.id,
                display_name=self.display_name,
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        else:  # openai
            return OpenAIPlayer(
                model=self.id,
                display_name=self.display_name,
                api_key=os.environ.get("OPENAI_API_KEY"),
            )


MODELS: list[ModelSpec] = [
    ModelSpec("gpt-4.1-mini", "GPT-4.1 Mini", "openai"),
    ModelSpec("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "anthropic"),
    ModelSpec("google/gemini-3-flash-preview", "Gemini 3 Flash", "openrouter"),
    ModelSpec("x-ai/grok-4.1-fast", "Grok 4.1 Fast", "openrouter"),
    ModelSpec("qwen/qwen3.5-flash-02-23", "Qwen 3.5 Flash", "openrouter"),
    ModelSpec("z-ai/glm-4.7-flash", "GLM 4.7 Flash", "openrouter"),
]
