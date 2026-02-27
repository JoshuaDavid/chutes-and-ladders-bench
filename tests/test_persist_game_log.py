"""Tests for _persist_game_log end-to-end wiring."""

from __future__ import annotations

import sqlite3
import json
import tempfile
from pathlib import Path

from chutes_bench.game import GameResult
from chutes_bench.invocation import LLMInvocation
from chutes_bench.persistence import ResultsDB
from chutes_bench.__main__ import _persist_game_log


def _make_db() -> tuple[ResultsDB, Path]:
    tmp = tempfile.mkdtemp()
    p = Path(tmp) / "test.db"
    return ResultsDB(p), p


def test_persist_game_log_writes_all_tables():
    db, path = _make_db()
    ma = db.ensure_model("m-a", "A", "openai")
    mb = db.ensure_model("m-b", "B", "openai")
    game_id = db.record_game_detailed(
        player_a_model_id=ma, player_b_model_id=mb,
        winner_idx=1, reason="forfeit", total_turns=1,
        max_turns_limit=200, system_prompt="test",
    )

    inv = LLMInvocation(
        request_messages=[{"role": "user", "content": "Your turn."}],
        response_raw={"choices": []},
        model_api_id="m-a",
        input_tokens=50,
        output_tokens=20,
        latency_ms=100,
    )

    result = GameResult(
        winner=1,
        reason="forfeit",
        turns=1,
        log=[
            {
                "turn_number": 1,
                "player": 0,
                "tool": "forfeit",
                "args": {},
                "board_before": [0, 0],
                "board_after": [0, 0],
                "result_ok": True,
                "result_message": "Player forfeits.",
                "is_winning_move": False,
                "is_illegal": False,
                "is_forfeit": True,
                "is_turn_over": False,
                "invocation": inv,
            }
        ],
    )

    _persist_game_log(db, game_id, result)

    conn = sqlite3.connect(path)

    # Check turns table
    turns = conn.execute("SELECT * FROM turns WHERE game_id = ?", (game_id,)).fetchall()
    assert len(turns) == 1

    # Check llm_invocations table
    invocations = conn.execute("SELECT * FROM llm_invocations WHERE game_id = ?", (game_id,)).fetchall()
    assert len(invocations) == 1

    # Check tool_calls table
    tool_calls = conn.execute("SELECT * FROM tool_calls WHERE game_id = ?", (game_id,)).fetchall()
    assert len(tool_calls) == 1

    conn.close()


def test_persist_game_log_multi_action_turn():
    """A turn with spin + move + end_turn produces 3 tool_calls but 1 turn."""
    db, path = _make_db()
    ma = db.ensure_model("m-a", "A", "openai")
    mb = db.ensure_model("m-b", "B", "openai")
    game_id = db.record_game_detailed(
        player_a_model_id=ma, player_b_model_id=mb,
        winner_idx=None, reason="max_turns", total_turns=1,
        max_turns_limit=1, system_prompt="test",
    )

    log = [
        {
            "turn_number": 1, "player": 0,
            "tool": "spin_spinner", "args": {},
            "board_before": [0, 0], "board_after": [0, 0],
            "result_ok": True, "result_message": "You spun a 3.",
            "is_winning_move": False, "is_illegal": False,
            "is_forfeit": False, "is_turn_over": False,
            "spin_value": 3,
        },
        {
            "turn_number": 1, "player": 0,
            "tool": "move_pawn_to_square", "args": {"square": 3},
            "board_before": [0, 0], "board_after": [0, 0],
            "result_ok": True, "result_message": "Moved to 3.",
            "is_winning_move": False, "is_illegal": False,
            "is_forfeit": False, "is_turn_over": False,
        },
        {
            "turn_number": 1, "player": 0,
            "tool": "end_turn", "args": {},
            "board_before": [0, 0], "board_after": [3, 0],
            "result_ok": True, "result_message": "Turn over.",
            "is_winning_move": False, "is_illegal": False,
            "is_forfeit": False, "is_turn_over": True,
        },
    ]

    result = GameResult(winner=None, reason="max_turns", turns=1, log=log)
    _persist_game_log(db, game_id, result)

    conn = sqlite3.connect(path)
    turns = conn.execute("SELECT * FROM turns WHERE game_id = ?", (game_id,)).fetchall()
    assert len(turns) == 1

    tool_calls = conn.execute("SELECT * FROM tool_calls WHERE game_id = ?", (game_id,)).fetchall()
    assert len(tool_calls) == 3

    invocations = conn.execute("SELECT * FROM llm_invocations WHERE game_id = ?", (game_id,)).fetchall()
    assert len(invocations) == 3

    # Spin value should be extracted
    conn.row_factory = sqlite3.Row
    turn_row = conn.execute("SELECT spin_value FROM turns WHERE game_id = ?", (game_id,)).fetchone()
    assert turn_row["spin_value"] == 3

    conn.close()
