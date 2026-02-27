"""LLM player implementations — OpenAI, Anthropic, and OpenRouter."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from chutes_bench.board import CHUTES_LADDERS
from chutes_bench.tools import TOOL_SCHEMAS

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

    @property
    def name(self) -> str:
        return self.display_name

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
        self._messages.append({"role": "user", "content": observation})

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            tools=TOOL_SCHEMAS,
            tool_choice="required",
        )
        msg = response.choices[0].message

        # Append assistant message to history
        self._messages.append(msg.model_dump(exclude_none=True))

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

    @property
    def name(self) -> str:
        return self.display_name

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def next_action(self, observation: str) -> tuple[str, dict]:
        self._messages.append({"role": "user", "content": observation})

        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=self._messages,
            tools=_openai_tools_to_anthropic(TOOL_SCHEMAS),
            tool_choice={"type": "any"},
        )

        # Build assistant message for history
        self._messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "tool_use":
                self._last_tool_use_id = block.id
                return block.name, block.input
        return "forfeit", {}

    def observe(self, message: str) -> None:
        self._messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": self._last_tool_use_id, "content": message}],
        })

    def reset(self) -> None:
        self._messages = []


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
