"""Tests for chutes_bench.tools (tool schemas + action validation)."""

from chutes_bench.board import BoardState
from chutes_bench.tools import (
    TOOL_SCHEMAS,
    validate_action,
    ActionResult,
    TurnPhase,
)


# ── tool schemas ─────────────────────────────────────────────────────

def test_tool_schemas_is_list():
    assert isinstance(TOOL_SCHEMAS, list)


def test_all_tools_present():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    expected = {
        "spin_spinner",
        "move_pawn_to_square",
        "ascend_ladder_to_square",
        "descend_chute_to_square",
        "end_turn",
        "send_message",
        "forfeit",
        "offer_draw",
        "accept_draw",
        "plan",
    }
    assert names == expected


# ── happy-path turn sequence ─────────────────────────────────────────

def test_spin_then_move_then_end():
    """Normal turn: spin → move → end_turn."""
    board = BoardState(positions=[10, 0])
    phase = TurnPhase(start_position=10)

    r1 = validate_action(board, player=0, tool_name="spin_spinner", args={}, phase=phase)
    assert r1.ok
    assert r1.spin_value is not None

    dest = 10 + r1.spin_value
    r2 = validate_action(
        board, player=0,
        tool_name="move_pawn_to_square",
        args={"square": dest},
        phase=phase,
    )
    assert r2.ok

    # If we landed on a chute/ladder, take it before ending
    if r2.requires_ladder:
        from chutes_bench.board import CHUTES_LADDERS
        r_cl = validate_action(
            board, player=0,
            tool_name="ascend_ladder_to_square",
            args={"square": CHUTES_LADDERS[dest]},
            phase=phase,
        )
        assert r_cl.ok
    elif r2.requires_chute:
        from chutes_bench.board import CHUTES_LADDERS
        r_cl = validate_action(
            board, player=0,
            tool_name="descend_chute_to_square",
            args={"square": CHUTES_LADDERS[dest]},
            phase=phase,
        )
        assert r_cl.ok

    r3 = validate_action(board, player=0, tool_name="end_turn", args={}, phase=phase)
    assert r3.ok
    assert r3.turn_over


def test_spin_required_before_move():
    board = BoardState(positions=[10, 0])
    phase = TurnPhase(start_position=10)
    r = validate_action(
        board, player=0,
        tool_name="move_pawn_to_square",
        args={"square": 14},
        phase=phase,
    )
    assert not r.ok


def test_must_move_to_correct_square():
    board = BoardState(positions=[10, 0])
    phase = TurnPhase(start_position=10)
    r1 = validate_action(board, player=0, tool_name="spin_spinner", args={}, phase=phase)
    correct = 10 + r1.spin_value
    wrong = correct + 1 if correct + 1 <= 100 else correct - 1
    r2 = validate_action(
        board, player=0,
        tool_name="move_pawn_to_square",
        args={"square": wrong},
        phase=phase,
    )
    assert not r2.ok


# ── chute/ladder actions ─────────────────────────────────────────────

def test_must_ascend_ladder_when_on_ladder_square():
    """Landing on a ladder base: must call ascend_ladder_to_square."""
    board = BoardState(positions=[0, 0])
    phase = TurnPhase(start_position=0)
    phase.has_spun = True
    phase.spin_value = 1  # 0+1=1 → ladder to 38

    r_move = validate_action(
        board, player=0,
        tool_name="move_pawn_to_square",
        args={"square": 1},
        phase=phase,
    )
    assert r_move.ok
    assert r_move.requires_ladder

    r_ladder = validate_action(
        board, player=0,
        tool_name="ascend_ladder_to_square",
        args={"square": 38},
        phase=phase,
    )
    assert r_ladder.ok


def test_must_descend_chute_when_on_chute_square():
    board = BoardState(positions=[10, 0])
    phase = TurnPhase(start_position=10)
    phase.spin_value = 6  # 10+6=16 → chute to 6
    phase.has_spun = True

    r_move = validate_action(
        board, player=0,
        tool_name="move_pawn_to_square",
        args={"square": 16},
        phase=phase,
    )
    assert r_move.ok
    assert r_move.requires_chute

    r_chute = validate_action(
        board, player=0,
        tool_name="descend_chute_to_square",
        args={"square": 6},
        phase=phase,
    )
    assert r_chute.ok


# ── forfeit ──────────────────────────────────────────────────────────

def test_forfeit_ends_game():
    board = BoardState(positions=[10, 0])
    phase = TurnPhase(start_position=10)
    r = validate_action(board, player=0, tool_name="forfeit", args={}, phase=phase)
    assert r.ok
    assert r.forfeit


# ── send_message is always legal ─────────────────────────────────────

def test_send_message_ok_anytime():
    board = BoardState(positions=[10, 0])
    phase = TurnPhase(start_position=10)
    r = validate_action(
        board, player=0,
        tool_name="send_message",
        args={"message": "good luck!"},
        phase=phase,
    )
    assert r.ok
    assert not r.turn_over


# ── bounce (overshoot) ───────────────────────────────────────────────

# ── plan ────────────────────────────────────────────────────────────

def test_plan_ok_before_spin():
    """plan is always allowed, even before spinning."""
    board = BoardState(positions=[10, 0])
    phase = TurnPhase(start_position=10)
    r = validate_action(
        board, player=0,
        tool_name="plan",
        args={"thought": "I should spin first, then move forward."},
        phase=phase,
    )
    assert r.ok
    assert not r.turn_over


def test_plan_ok_after_spin():
    """plan is allowed mid-turn after spinning."""
    board = BoardState(positions=[10, 0])
    phase = TurnPhase(start_position=10)
    phase.has_spun = True
    phase.spin_value = 3
    r = validate_action(
        board, player=0,
        tool_name="plan",
        args={"thought": "I spun 3, so I should move to 13."},
        phase=phase,
    )
    assert r.ok
    assert not r.turn_over


def test_plan_does_not_change_turn_phase():
    """plan must not mutate TurnPhase."""
    board = BoardState(positions=[10, 0])
    phase = TurnPhase(start_position=10)
    validate_action(board, player=0, tool_name="plan", args={"thought": "hmm"}, phase=phase)
    assert not phase.has_spun
    assert phase.current_position is None
    assert not phase.reached_final


# ── bounce (overshoot) ───────────────────────────────────────────────

def test_overshoot_spin_still_needs_end_turn():
    """Spin that overshoots 100: player stays, must end turn."""
    board = BoardState(positions=[96, 0])
    phase = TurnPhase(start_position=96)
    phase.spin_value = 5  # 96+5=101 → bounce
    phase.has_spun = True

    r_move = validate_action(
        board, player=0,
        tool_name="move_pawn_to_square",
        args={"square": 96},  # stay put
        phase=phase,
    )
    assert r_move.ok
    assert r_move.bounced
