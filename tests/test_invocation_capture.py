"""RED — tests that LLM players capture invocation metadata."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from chutes_bench.invocation import LLMInvocation
from chutes_bench.players import OpenAIPlayer, AnthropicPlayer


# ── helpers ──────────────────────────────────────────────────────────

def _make_openai_response(*tool_calls, input_tokens=10, output_tokens=5):
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
    msg.model_dump = lambda exclude_none=False: {
        "role": "assistant",
        "tool_calls": [
            {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tcs
        ],
    }
    usage = SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=usage,
        model_dump=lambda: {
            "choices": [{"message": msg.model_dump()}],
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
        },
    )


def _make_anthropic_response(*tool_uses, input_tokens=12, output_tokens=8):
    content = []
    for i, (name, inp) in enumerate(tool_uses):
        content.append(SimpleNamespace(type="tool_use", id=f"toolu_{i}", name=name, input=inp))
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    resp = SimpleNamespace(
        content=content,
        usage=usage,
        model_dump=lambda: {
            "content": [{"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": inp}
                        for i, (name, inp) in enumerate(tool_uses)],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    )
    return resp


# ── OpenAI player ────────────────────────────────────────────────────

def test_openai_player_captures_last_invocation():
    player = OpenAIPlayer(model="gpt-test", display_name="Test")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(
        ("spin_spinner", {}),
    )
    player._client = mock_client

    player.next_action("Your turn.")

    inv = player.last_invocation
    assert inv is not None
    assert isinstance(inv, LLMInvocation)
    assert inv.model_api_id == "gpt-test"
    assert inv.input_tokens == 10
    assert inv.output_tokens == 5
    assert len(inv.request_messages) > 0
    assert inv.request_messages[0]["role"] == "system"
    assert isinstance(inv.response_raw, dict)


def test_openai_player_invocation_has_latency():
    player = OpenAIPlayer(model="gpt-test", display_name="Test")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(
        ("spin_spinner", {}),
    )
    player._client = mock_client

    player.next_action("Your turn.")

    inv = player.last_invocation
    assert inv is not None
    assert inv.latency_ms is not None
    assert inv.latency_ms >= 0


def test_openai_player_request_messages_are_snapshot():
    """request_messages should be a snapshot at call time, not a live reference."""
    player = OpenAIPlayer(model="gpt-test", display_name="Test")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _make_openai_response(("spin_spinner", {})),
        _make_openai_response(("end_turn", {})),
    ]
    player._client = mock_client

    player.next_action("Your turn.")
    first_msgs = player.last_invocation.request_messages

    player.observe("You spun a 4.")
    player.next_action("Continue.")
    second_msgs = player.last_invocation.request_messages

    # First snapshot should be shorter than second (more messages accumulated)
    assert len(first_msgs) < len(second_msgs)


# ── Anthropic player ────────────────────────────────────────────────

def test_anthropic_player_captures_last_invocation():
    player = AnthropicPlayer(model="claude-test", display_name="Test")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_response(
        ("spin_spinner", {}),
    )
    player._client = mock_client

    player.next_action("Your turn.")

    inv = player.last_invocation
    assert inv is not None
    assert isinstance(inv, LLMInvocation)
    assert inv.model_api_id == "claude-test"
    assert inv.input_tokens == 12
    assert inv.output_tokens == 8
    assert len(inv.request_messages) > 0
    assert isinstance(inv.response_raw, dict)


def test_anthropic_player_invocation_has_latency():
    player = AnthropicPlayer(model="claude-test", display_name="Test")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_response(
        ("spin_spinner", {}),
    )
    player._client = mock_client

    player.next_action("Your turn.")

    inv = player.last_invocation
    assert inv is not None
    assert inv.latency_ms is not None
    assert inv.latency_ms >= 0
