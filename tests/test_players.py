"""Tests for LLM player implementations."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from chutes_bench.players import AnthropicPlayer, OpenAIPlayer


def _make_response(*tool_calls):
    """Build a fake OpenAI ChatCompletion response with given tool calls."""
    tcs = []
    for i, (name, args) in enumerate(tool_calls):
        tcs.append(SimpleNamespace(
            id=f"call_{i}",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        ))
    msg = SimpleNamespace(
        tool_calls=tcs,
        content=None,
        role="assistant",
    )
    # model_dump returns a plain dict (matches real openai SDK behavior)
    msg.model_dump = lambda exclude_none=False: {
        "role": "assistant",
        "tool_calls": [
            {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tcs
        ],
    }
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_multi_tool_call_only_keeps_first_in_history():
    """When the model returns multiple tool_calls, only the first should be
    kept in the conversation history. Otherwise the OpenAI API rejects the
    next request because subsequent tool_call_ids lack responses."""
    player = OpenAIPlayer(model="test", display_name="Test")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_response(
        ("spin_spinner", {}),
        ("move_pawn_to_square", {"square": 5}),
    )
    player._client = mock_client

    tool_name, args = player.next_action("Your turn.")
    assert tool_name == "spin_spinner"

    # The assistant message saved in history should have only one tool call
    assistant_msg = player._messages[-1]
    assert len(assistant_msg["tool_calls"]) == 1
    assert assistant_msg["tool_calls"][0]["id"] == "call_0"


def _make_anthropic_response(*tool_uses):
    """Build a fake Anthropic Messages response with given tool_use blocks."""
    content = []
    for i, (name, inp) in enumerate(tool_uses):
        content.append(SimpleNamespace(type="tool_use", id=f"toolu_{i}", name=name, input=inp))
    return SimpleNamespace(content=content)


def test_anthropic_multi_tool_use_only_keeps_first_in_history():
    """When Anthropic returns multiple tool_use blocks, only the first should
    be kept in the conversation history so every tool_use gets a tool_result."""
    player = AnthropicPlayer(model="test", display_name="Test")

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_response(
        ("spin_spinner", {}),
        ("move_pawn_to_square", {"square": 5}),
    )
    player._client = mock_client

    tool_name, args = player.next_action("Your turn.")
    assert tool_name == "spin_spinner"

    # The assistant message saved in history should have only one tool_use block
    assistant_msg = player._messages[-1]
    tool_use_blocks = [b for b in assistant_msg["content"] if getattr(b, "type", None) == "tool_use"]
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0].id == "toolu_0"


def test_openai_next_action_adds_tool_result_for_orphaned_tool_call():
    """If next_action is called when the last message is an assistant with
    an unresolved tool_call (no observe() was called), it should inject a
    synthetic tool result before adding the new user observation. This happens
    when the game loop ends a turn without calling observe()."""
    player = OpenAIPlayer(model="test", display_name="Test")

    mock_client = MagicMock()
    # First call: end_turn
    mock_client.chat.completions.create.side_effect = [
        _make_response(("end_turn", {})),
        _make_response(("spin_spinner", {})),
    ]
    player._client = mock_client

    # First action returns end_turn
    tool_name, _ = player.next_action("Your turn. Square 5.")
    assert tool_name == "end_turn"
    # Simulate game loop NOT calling observe() — turn is over

    # Second action for next turn — should NOT crash
    tool_name, _ = player.next_action("Your turn. Square 5.")
    assert tool_name == "spin_spinner"

    # Verify the history has a tool result between the two assistant messages
    roles = [m.get("role") for m in player._messages]
    # Should be: system, user, assistant, tool, user, assistant
    assert roles == ["system", "user", "assistant", "tool", "user", "assistant"]
