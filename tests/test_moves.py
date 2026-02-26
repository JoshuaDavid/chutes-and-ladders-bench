"""Tests for flexible multi-step move sequences and first-message observation."""

from chutes_bench.board import BoardState
from chutes_bench.tools import TurnPhase, validate_action


# ── First message ────────────────────────────────────────────────────

def test_first_observation_for_player_a():
    """The very first message player A receives at game start."""
    from chutes_bench.game import GameRunner

    observations: list[str] = []

    def capture_next_action(observation: str) -> tuple[str, dict]:
        observations.append(observation)
        return ("forfeit", {})

    p0 = _make_fake_player("A", capture_next_action)
    p1 = _make_fake_player("B", lambda obs: ("forfeit", {}))

    runner = GameRunner(players=[p0, p1], max_turns=10)
    runner.play()

    assert len(observations) >= 1
    first = observations[0]
    assert "not yet on the board" in first
    assert "square 0" in first or "not yet on the board" in first


# ── ALLOWED multi-step sequences ─────────────────────────────────────
# Player on square 0, spins 4 → landing square 4 (ladder to 14)

def test_allowed_direct_move_then_ladder():
    """move(4), ascend_ladder(14), end_turn — standard flow."""
    board = BoardState(positions=[0, 0])
    phase = TurnPhase()
    phase.has_spun = True
    phase.spin_value = 4
    phase.start_position = 0

    r1 = validate_action(board, 0, "move_pawn_to_square", {"square": 4}, phase)
    assert r1.ok, r1.message

    r2 = validate_action(board, 0, "ascend_ladder_to_square", {"square": 14}, phase)
    assert r2.ok, r2.message

    r3 = validate_action(board, 0, "end_turn", {}, phase)
    assert r3.ok, r3.message
    assert r3.turn_over


def test_allowed_skip_to_ladder_dest():
    """move(14), end_turn — jump straight to ladder destination."""
    board = BoardState(positions=[0, 0])
    phase = TurnPhase()
    phase.has_spun = True
    phase.spin_value = 4
    phase.start_position = 0

    r1 = validate_action(board, 0, "move_pawn_to_square", {"square": 14}, phase)
    assert r1.ok, r1.message

    r2 = validate_action(board, 0, "end_turn", {}, phase)
    assert r2.ok, r2.message
    assert r2.turn_over


def test_allowed_step_by_step_then_ladder():
    """move(1), move(2), move(3), move(4), ascend_ladder(14), end_turn."""
    board = BoardState(positions=[0, 0])
    phase = TurnPhase()
    phase.has_spun = True
    phase.spin_value = 4
    phase.start_position = 0

    for sq in [1, 2, 3, 4]:
        r = validate_action(board, 0, "move_pawn_to_square", {"square": sq}, phase)
        assert r.ok, f"move({sq}) failed: {r.message}"

    r_ladder = validate_action(board, 0, "ascend_ladder_to_square", {"square": 14}, phase)
    assert r_ladder.ok, r_ladder.message

    r_end = validate_action(board, 0, "end_turn", {}, phase)
    assert r_end.ok, r_end.message
    assert r_end.turn_over


def test_allowed_partial_steps_then_jump():
    """move(2), move(4), ascend_ladder(14), end_turn — skip square 3."""
    board = BoardState(positions=[0, 0])
    phase = TurnPhase()
    phase.has_spun = True
    phase.spin_value = 4
    phase.start_position = 0

    r1 = validate_action(board, 0, "move_pawn_to_square", {"square": 2}, phase)
    assert r1.ok, r1.message

    r2 = validate_action(board, 0, "move_pawn_to_square", {"square": 4}, phase)
    assert r2.ok, r2.message

    r3 = validate_action(board, 0, "ascend_ladder_to_square", {"square": 14}, phase)
    assert r3.ok, r3.message

    r4 = validate_action(board, 0, "end_turn", {}, phase)
    assert r4.ok, r4.message


# ── NOT ALLOWED sequences ───────────────────────────────────────────
# Player on square 0, spins 4

def test_not_allowed_end_turn_without_moving():
    """end_turn without any move → illegal."""
    board = BoardState(positions=[0, 0])
    phase = TurnPhase()
    phase.has_spun = True
    phase.spin_value = 4
    phase.start_position = 0

    r = validate_action(board, 0, "end_turn", {}, phase)
    assert not r.ok
    assert r.illegal


def test_not_allowed_move_to_wrong_square():
    """move(5) when spin is 4 (from 0) → illegal."""
    board = BoardState(positions=[0, 0])
    phase = TurnPhase()
    phase.has_spun = True
    phase.spin_value = 4
    phase.start_position = 0

    r = validate_action(board, 0, "move_pawn_to_square", {"square": 5}, phase)
    assert not r.ok
    assert r.illegal


def test_not_allowed_stop_at_intermediate():
    """move(1), end_turn → can't stop at intermediate square."""
    board = BoardState(positions=[0, 0])
    phase = TurnPhase()
    phase.has_spun = True
    phase.spin_value = 4
    phase.start_position = 0

    r1 = validate_action(board, 0, "move_pawn_to_square", {"square": 1}, phase)
    assert r1.ok  # valid intermediate move

    r2 = validate_action(board, 0, "end_turn", {}, phase)
    assert not r2.ok
    assert r2.illegal


def test_not_allowed_stall_with_messages():
    """move(1), then 19 send_messages doesn't complete the turn.
    After MAX actions, the game runner should end the turn as illegal."""
    board = BoardState(positions=[0, 0])
    phase = TurnPhase()
    phase.has_spun = True
    phase.spin_value = 4
    phase.start_position = 0

    r1 = validate_action(board, 0, "move_pawn_to_square", {"square": 1}, phase)
    assert r1.ok

    # send_message is always OK but doesn't advance the turn
    for _ in range(19):
        r = validate_action(board, 0, "send_message", {"message": "stalling"}, phase)
        assert r.ok

    # end_turn should still be illegal because we're at intermediate square 1
    r_end = validate_action(board, 0, "end_turn", {}, phase)
    assert not r_end.ok
    assert r_end.illegal


# Player on square 0, spins 5 (target = 5, no chute/ladder)

def test_not_allowed_ladder_from_intermediate():
    """Spin 5 (target=5), move(4), ascend_ladder(14) → illegal.
    Square 4 has a ladder but it's not the landing square."""
    board = BoardState(positions=[0, 0])
    phase = TurnPhase()
    phase.has_spun = True
    phase.spin_value = 5
    phase.start_position = 0

    r1 = validate_action(board, 0, "move_pawn_to_square", {"square": 4}, phase)
    assert r1.ok  # valid intermediate

    r2 = validate_action(board, 0, "ascend_ladder_to_square", {"square": 14}, phase)
    assert not r2.ok
    assert r2.illegal


def test_not_allowed_move_backward():
    """move(3), move(2) → can't go backward."""
    board = BoardState(positions=[0, 0])
    phase = TurnPhase()
    phase.has_spun = True
    phase.spin_value = 4
    phase.start_position = 0

    r1 = validate_action(board, 0, "move_pawn_to_square", {"square": 3}, phase)
    assert r1.ok

    r2 = validate_action(board, 0, "move_pawn_to_square", {"square": 2}, phase)
    assert not r2.ok
    assert r2.illegal


# ── Chute multi-step ─────────────────────────────────────────────────

def test_allowed_direct_to_chute_dest():
    """From square 10, spin 6, target=16 (chute to 6). move(6), end_turn."""
    board = BoardState(positions=[10, 0])
    phase = TurnPhase()
    phase.has_spun = True
    phase.spin_value = 6
    phase.start_position = 10

    r1 = validate_action(board, 0, "move_pawn_to_square", {"square": 6}, phase)
    assert r1.ok, r1.message

    r2 = validate_action(board, 0, "end_turn", {}, phase)
    assert r2.ok, r2.message


def test_allowed_move_to_chute_then_descend():
    """From 10, spin 6: move(16), descend_chute(6), end_turn."""
    board = BoardState(positions=[10, 0])
    phase = TurnPhase()
    phase.has_spun = True
    phase.spin_value = 6
    phase.start_position = 10

    r1 = validate_action(board, 0, "move_pawn_to_square", {"square": 16}, phase)
    assert r1.ok, r1.message

    r2 = validate_action(board, 0, "descend_chute_to_square", {"square": 6}, phase)
    assert r2.ok, r2.message

    r3 = validate_action(board, 0, "end_turn", {}, phase)
    assert r3.ok, r3.message


# ── Helpers ──────────────────────────────────────────────────────────

def _make_fake_player(name: str, next_action_fn):
    """Create a fake player using composition (not inheritance)."""

    class _Fake:
        def __init__(self):
            self._name = name

        @property
        def name(self) -> str:
            return self._name

        def next_action(self, observation: str) -> tuple[str, dict]:
            return next_action_fn(observation)

        def observe(self, message: str) -> None:
            pass

    return _Fake()
