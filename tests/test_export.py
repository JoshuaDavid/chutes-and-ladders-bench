"""Tests for game event export."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from chutes_bench.persistence import ResultsDB


@pytest.fixture
def populated_db(tmp_path: Path) -> Path:
    """Create a test DB with known game data."""
    db_path = tmp_path / "test.db"
    db = ResultsDB(db_path)

    # Record a game using the detailed API
    ma = db.ensure_model("model-a", "Model A", "openai")
    mb = db.ensure_model("model-b", "Model B", "anthropic")
    game_id = db.record_game_detailed(
        player_a_model_id=ma,
        player_b_model_id=mb,
        winner_idx=0,
        reason="win",
        total_turns=2,
        max_turns_limit=200,
        system_prompt="test",
    )

    # Turn 1, player 0: spin + move + end_turn
    inv1 = db.record_llm_invocation(
        game_id=game_id, turn_number=1, player_idx=0,
        sequence_in_turn=0, model_api_id="model-a",
        request_messages=[{"role": "user", "content": "Your turn."}],
        response_raw={"choices": [{"message": {"content": "I'll spin!", "tool_calls": []}}]},
        input_tokens=100, output_tokens=20, latency_ms=500,
    )
    db.record_tool_call(
        invocation_id=inv1, game_id=game_id, turn_number=1, player_idx=0,
        tool_name="spin_spinner", tool_args={},
        result_ok=True, result_message="You spun a 3.",
        board_before=[0, 0], board_after=[0, 0],
    )
    db.record_tool_call(
        invocation_id=inv1, game_id=game_id, turn_number=1, player_idx=0,
        tool_name="move_pawn_to_square", tool_args={"square": 3},
        result_ok=True, result_message="Moved to square 3.",
        board_before=[0, 0], board_after=[3, 0],
    )

    inv2 = db.record_llm_invocation(
        game_id=game_id, turn_number=1, player_idx=0,
        sequence_in_turn=1, model_api_id="model-a",
        request_messages=[], response_raw={},
    )
    db.record_tool_call(
        invocation_id=inv2, game_id=game_id, turn_number=1, player_idx=0,
        tool_name="end_turn", tool_args={},
        result_ok=True, result_message="Turn ended.",
        board_before=[3, 0], board_after=[3, 0],
        is_turn_over=True,
    )

    db.record_turn(
        game_id=game_id, turn_number=1, player_idx=0,
        start_position=0, end_position=3, spin_value=3,
        outcome="normal", actions_count=3,
    )

    # Turn 2, player 1: spin + move + end_turn
    inv3 = db.record_llm_invocation(
        game_id=game_id, turn_number=2, player_idx=1,
        sequence_in_turn=0, model_api_id="model-b",
        request_messages=[{"role": "user", "content": "Your turn."}],
        response_raw={"content": [{"type": "text", "text": "Let me spin."}]},
        input_tokens=80, output_tokens=15, latency_ms=400,
    )
    db.record_tool_call(
        invocation_id=inv3, game_id=game_id, turn_number=2, player_idx=1,
        tool_name="spin_spinner", tool_args={},
        result_ok=True, result_message="You spun a 5.",
        board_before=[3, 0], board_after=[3, 0],
    )
    db.record_tool_call(
        invocation_id=inv3, game_id=game_id, turn_number=2, player_idx=1,
        tool_name="move_pawn_to_square", tool_args={"square": 5},
        result_ok=True, result_message="Moved to square 5.",
        board_before=[3, 0], board_after=[3, 5],
    )

    inv4 = db.record_llm_invocation(
        game_id=game_id, turn_number=2, player_idx=1,
        sequence_in_turn=1, model_api_id="model-b",
        request_messages=[], response_raw={},
    )
    db.record_tool_call(
        invocation_id=inv4, game_id=game_id, turn_number=2, player_idx=1,
        tool_name="end_turn", tool_args={},
        result_ok=True, result_message="Turn ended.",
        board_before=[3, 5], board_after=[3, 5],
        is_turn_over=True,
    )

    db.record_turn(
        game_id=game_id, turn_number=2, player_idx=1,
        start_position=0, end_position=5, spin_value=5,
        outcome="normal", actions_count=3,
    )

    db.close()
    return db_path


def test_export_games_list(populated_db: Path) -> None:
    from chutes_bench.export import export_games_list

    games = export_games_list(populated_db)
    assert len(games) == 1
    game = games[0]
    assert game["id"] == 1
    assert game["player_a"] == "Model A"
    assert game["player_b"] == "Model B"
    assert game["winner"] == "Model A"
    assert game["reason"] == "win"
    assert game["turns"] == 2


def test_export_games_list_empty(tmp_path: Path) -> None:
    from chutes_bench.export import export_games_list

    db_path = tmp_path / "empty.db"
    db = ResultsDB(db_path)
    db.close()
    games = export_games_list(db_path)
    assert games == []


def test_export_game_events_returns_game_metadata(populated_db: Path) -> None:
    from chutes_bench.export import export_game_events

    result = export_game_events(populated_db, game_id=1)
    assert result is not None
    assert result["game"]["id"] == 1
    assert result["game"]["player_a"] == "Model A"
    assert result["game"]["player_b"] == "Model B"
    assert result["game"]["winner"] == "Model A"
    assert result["game"]["reason"] == "win"


def test_export_game_events_has_turns(populated_db: Path) -> None:
    from chutes_bench.export import export_game_events

    result = export_game_events(populated_db, game_id=1)
    assert result is not None
    turns = result["turns"]
    assert len(turns) == 2
    assert turns[0]["turn_number"] == 1
    assert turns[0]["player_idx"] == 0
    assert turns[0]["start_position"] == 0
    assert turns[0]["end_position"] == 3
    assert turns[0]["spin_value"] == 3
    assert turns[0]["outcome"] == "normal"
    assert turns[1]["turn_number"] == 2
    assert turns[1]["player_idx"] == 1


def test_export_game_events_has_tool_calls(populated_db: Path) -> None:
    from chutes_bench.export import export_game_events

    result = export_game_events(populated_db, game_id=1)
    assert result is not None
    events = result["events"]

    tool_events = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_events) == 6  # 3 per turn, 2 turns

    spin = tool_events[0]
    assert spin["tool_name"] == "spin_spinner"
    assert spin["result_ok"] is True
    assert spin["result_message"] == "You spun a 3."
    assert spin["board_before"] == [0, 0]
    assert spin["board_after"] == [0, 0]
    assert spin["turn_number"] == 1
    assert spin["player_idx"] == 0


def test_export_game_events_has_llm_responses(populated_db: Path) -> None:
    from chutes_bench.export import export_game_events

    result = export_game_events(populated_db, game_id=1)
    assert result is not None
    events = result["events"]

    llm_events = [e for e in events if e["type"] == "llm_response"]
    # Only invocations with text content become llm_response events
    assert len(llm_events) == 2  # "I'll spin!" (OpenAI) and "Let me spin." (Anthropic)
    assert llm_events[0]["text"] == "I'll spin!"
    assert llm_events[0]["model"] == "model-a"
    assert llm_events[1]["text"] == "Let me spin."
    assert llm_events[1]["model"] == "model-b"


def test_export_game_events_chronological_order(populated_db: Path) -> None:
    from chutes_bench.export import export_game_events

    result = export_game_events(populated_db, game_id=1)
    assert result is not None
    events = result["events"]

    # Events should be in chronological order: turn 1 before turn 2
    turn_numbers = [e["turn_number"] for e in events]
    assert turn_numbers == sorted(turn_numbers)

    # Within turn 1: llm_response, then tool calls
    turn1 = [e for e in events if e["turn_number"] == 1]
    assert turn1[0]["type"] == "llm_response"
    assert turn1[1]["type"] == "tool_call"
    assert turn1[1]["tool_name"] == "spin_spinner"


def test_export_game_events_nonexistent(populated_db: Path) -> None:
    from chutes_bench.export import export_game_events

    result = export_game_events(populated_db, game_id=999)
    assert result is None


def test_export_game_events_has_token_totals(populated_db: Path) -> None:
    from chutes_bench.export import export_game_events

    result = export_game_events(populated_db, game_id=1)
    assert result is not None
    totals = result["token_totals"]
    # Player 0 (model-a): inv1 has 100 in + 20 out, inv2 has no tokens
    assert totals["player_a"]["input_tokens"] == 100
    assert totals["player_a"]["output_tokens"] == 20
    # Player 1 (model-b): inv3 has 80 in + 15 out, inv4 has no tokens
    assert totals["player_b"]["input_tokens"] == 80
    assert totals["player_b"]["output_tokens"] == 15


def test_export_game_events_json_serializable(populated_db: Path) -> None:
    from chutes_bench.export import export_game_events

    result = export_game_events(populated_db, game_id=1)
    assert result is not None
    # Should be fully JSON-serializable
    serialized = json.dumps(result)
    roundtripped = json.loads(serialized)
    assert roundtripped["game"]["id"] == 1
    assert len(roundtripped["events"]) > 0
