"""RED — tests for chutes_bench.tools (tool schemas + action validation)."""

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
    }
    assert names == expected


# ── happy-path turn sequence ─────────────────────────────────────────

def test_spin_then_move_then_end():
    """Normal turn: spin → move → end_turn."""
    board = BoardState(positions=[10, 0])
    phase = TurnPhase()

    r1 = validate_action(board, player=0, tool_name="spin_spinner", args={}, phase=phase)
    assert r1.ok
    assert r1.spin_value is not None

    dest = 10 + r1.spin_value
    # Skip if chute/ladder square — just test the flow
    r2 = validate_action(
        board, player=0,
        tool_name="move_pawn_to_square",
        args={"square": dest},
        phase=phase,
    )
    assert r2.ok

    r3 = validate_action(board, player=0, tool_name="end_turn", args={}, phase=phase)
    assert r3.ok
    assert r3.turn_over


def test_spin_required_before_move():
    board = BoardState(positions=[10, 0])
    phase = TurnPhase()
    r = validate_action(
        board, player=0,
        tool_name="move_pawn_to_square",
        args={"square": 14},
        phase=phase,
    )
    assert not r.ok
    assert r.illegal


def test_must_move_to_correct_square():
    board = BoardState(positions=[10, 0])
    phase = TurnPhase()
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
    assert r2.illegal


# ── chute/ladder actions ─────────────────────────────────────────────

def test_must_ascend_ladder_when_on_ladder_square():
    """Landing on a ladder base: must call ascend_ladder_to_square."""
    board = BoardState(positions=[0, 0])
    phase = TurnPhase()
    # Force spin=1 scenario: 0+1=1, which is a ladder to 38
    validate_action(board, player=0, tool_name="spin_spinner", args={}, phase=phase)
    # Pretend spin was 1 by overriding phase
    phase.spin_value = 1
    phase.has_spun = True
    phase.has_moved = False

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
    phase = TurnPhase()
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
    phase = TurnPhase()
    r = validate_action(board, player=0, tool_name="forfeit", args={}, phase=phase)
    assert r.ok
    assert r.forfeit


# ── send_message is always legal ─────────────────────────────────────

def test_send_message_ok_anytime():
    board = BoardState(positions=[10, 0])
    phase = TurnPhase()
    r = validate_action(
        board, player=0,
        tool_name="send_message",
        args={"message": "good luck!"},
        phase=phase,
    )
    assert r.ok
    assert not r.turn_over


# ── bounce (overshoot) ───────────────────────────────────────────────

def test_overshoot_spin_still_needs_end_turn():
    """Spin that overshoots 100: player stays, must end turn."""
    board = BoardState(positions=[96, 0])
    phase = TurnPhase()
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
