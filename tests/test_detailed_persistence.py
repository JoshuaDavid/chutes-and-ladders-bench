"""RED — tests for detailed game logging in persistence layer."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

from chutes_bench.persistence import ResultsDB


def _make_db() -> tuple[ResultsDB, Path]:
    tmp = tempfile.mkdtemp()
    p = Path(tmp) / "test.db"
    return ResultsDB(p), p


def test_models_table_exists():
    db, path = _make_db()
    conn = sqlite3.connect(path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "models" in tables


def test_ensure_model_inserts_and_deduplicates():
    db, _ = _make_db()
    mid1 = db.ensure_model("gpt-4.1-mini", "GPT-4.1 Mini", "openai")
    mid2 = db.ensure_model("gpt-4.1-mini", "GPT-4.1 Mini", "openai")
    assert mid1 == mid2
    assert isinstance(mid1, int)


def test_detailed_tables_exist():
    db, path = _make_db()
    conn = sqlite3.connect(path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    for t in ("models", "turns", "llm_invocations", "tool_calls"):
        assert t in tables, f"Missing table: {t}"


def test_record_game_detailed():
    db, path = _make_db()
    ma = db.ensure_model("gpt-4.1-mini", "GPT-4.1 Mini", "openai")
    mb = db.ensure_model("claude-haiku", "Claude Haiku", "anthropic")

    game_id = db.record_game_detailed(
        player_a_model_id=ma,
        player_b_model_id=mb,
        winner_idx=0,
        reason="win",
        total_turns=10,
        max_turns_limit=200,
        system_prompt="You are playing...",
    )
    assert isinstance(game_id, int)

    # Also records in legacy games table for Elo compat
    outcomes = db.list_outcomes()
    assert len(outcomes) == 1
    assert outcomes[0].winner == "GPT-4.1 Mini"


def test_record_turn():
    db, _ = _make_db()
    ma = db.ensure_model("m-a", "A", "openai")
    mb = db.ensure_model("m-b", "B", "openai")
    gid = db.record_game_detailed(
        player_a_model_id=ma, player_b_model_id=mb,
        winner_idx=None, reason="max_turns", total_turns=2,
        max_turns_limit=200, system_prompt="test",
    )

    tid = db.record_turn(
        game_id=gid, turn_number=1, player_idx=0,
        start_position=0, end_position=38,
        spin_value=1, outcome="normal", actions_count=4,
    )
    assert isinstance(tid, int)


def test_record_llm_invocation():
    db, _ = _make_db()
    ma = db.ensure_model("m-a", "A", "openai")
    mb = db.ensure_model("m-b", "B", "openai")
    gid = db.record_game_detailed(
        player_a_model_id=ma, player_b_model_id=mb,
        winner_idx=0, reason="win", total_turns=5,
        max_turns_limit=200, system_prompt="test",
    )

    inv_id = db.record_llm_invocation(
        game_id=gid, turn_number=1, player_idx=0,
        sequence_in_turn=0, model_api_id="m-a",
        request_messages=[{"role": "system", "content": "test"}],
        response_raw={"choices": []},
        input_tokens=100, output_tokens=50, latency_ms=320,
    )
    assert isinstance(inv_id, int)


def test_record_tool_call():
    db, _ = _make_db()
    ma = db.ensure_model("m-a", "A", "openai")
    mb = db.ensure_model("m-b", "B", "openai")
    gid = db.record_game_detailed(
        player_a_model_id=ma, player_b_model_id=mb,
        winner_idx=0, reason="win", total_turns=5,
        max_turns_limit=200, system_prompt="test",
    )
    inv_id = db.record_llm_invocation(
        game_id=gid, turn_number=1, player_idx=0,
        sequence_in_turn=0, model_api_id="m-a",
        request_messages=[], response_raw={},
        input_tokens=10, output_tokens=5, latency_ms=100,
    )

    tc_id = db.record_tool_call(
        invocation_id=inv_id, game_id=gid,
        turn_number=1, player_idx=0,
        tool_name="spin_spinner", tool_args={},
        result_ok=True, result_message="You spun a 4.",
        board_before=[0, 0], board_after=[0, 0],
        is_winning_move=False, is_illegal=False,
        is_forfeit=False, is_turn_over=False,
    )
    assert isinstance(tc_id, int)


def test_round_trip_json_fields():
    """JSON fields survive a round-trip through the database."""
    db, path = _make_db()
    ma = db.ensure_model("m-a", "A", "openai")
    mb = db.ensure_model("m-b", "B", "openai")
    gid = db.record_game_detailed(
        player_a_model_id=ma, player_b_model_id=mb,
        winner_idx=0, reason="win", total_turns=1,
        max_turns_limit=200, system_prompt="test",
    )
    msgs = [{"role": "system", "content": "hello"}]
    resp = {"choices": [{"message": {"tool_calls": []}}]}
    inv_id = db.record_llm_invocation(
        game_id=gid, turn_number=1, player_idx=0,
        sequence_in_turn=0, model_api_id="m-a",
        request_messages=msgs, response_raw=resp,
        input_tokens=10, output_tokens=5, latency_ms=100,
    )

    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT request_messages, response_raw FROM llm_invocations WHERE id = ?",
        (inv_id,),
    ).fetchone()
    assert json.loads(row[0]) == msgs
    assert json.loads(row[1]) == resp


def test_game_detailed_with_no_winner():
    """A draw/max_turns game records winner_idx as NULL."""
    db, _ = _make_db()
    ma = db.ensure_model("m-a", "A", "openai")
    mb = db.ensure_model("m-b", "B", "openai")
    gid = db.record_game_detailed(
        player_a_model_id=ma, player_b_model_id=mb,
        winner_idx=None, reason="draw", total_turns=10,
        max_turns_limit=200, system_prompt="test",
    )
    outcomes = db.list_outcomes()
    assert len(outcomes) == 1
    assert outcomes[0].winner is None
