"""Tests for chutes_bench.game (game runner)."""

from chutes_bench.board import BoardState
from chutes_bench.game import GameRunner, GameResult


class FakePlayer:
    """Deterministic player for testing — plays a scripted sequence of actions."""

    def __init__(self, script: list[tuple[str, dict]], player_name: str = "fake"):
        self.script = list(script)
        self._idx = 0
        self._name = player_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def last_invocation(self):
        return None

    def next_action(self, observation: str) -> tuple[str, dict]:
        action = self.script[self._idx]
        self._idx += 1
        return action

    def observe(self, message: str) -> None:
        pass


def test_forfeit_game():
    """Player 0 forfeits immediately → player 1 wins."""
    p0 = FakePlayer([("forfeit", {})])
    p1 = FakePlayer([])

    runner = GameRunner(players=[p0, p1], max_turns=10)
    result = runner.play()

    assert isinstance(result, GameResult)
    assert result.winner == 1
    assert result.reason == "forfeit"


def test_illegal_move_loses():
    """Player 0 tries to move without spinning → illegal → loss."""
    p0 = FakePlayer([("move_pawn_to_square", {"square": 5})])
    p1 = FakePlayer([])

    runner = GameRunner(players=[p0, p1], max_turns=10)
    result = runner.play()

    assert result.winner == 1
    assert result.reason == "illegal_move"


def test_max_turns_draw():
    """If max_turns exceeded, game is a draw."""
    # Both players just send messages forever (never spin/move)
    p0 = FakePlayer([("send_message", {"message": "hi"})] * 100)
    p1 = FakePlayer([("send_message", {"message": "hi"})] * 100)

    runner = GameRunner(players=[p0, p1], max_turns=4)
    result = runner.play()

    assert result.winner is None
    assert result.reason == "max_turns"
