"""RED — tests that GameRunner captures board state and invocation data in log."""

from chutes_bench.board import BoardState
from chutes_bench.game import GameRunner, GameResult
from chutes_bench.invocation import LLMInvocation


class FakePlayerWithInvocation:
    """Deterministic player that exposes a fake last_invocation."""

    def __init__(self, script: list[tuple[str, dict]], player_name: str = "fake"):
        self.script = list(script)
        self._idx = 0
        self._name = player_name
        self._last_invocation: LLMInvocation | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def last_invocation(self) -> LLMInvocation | None:
        return self._last_invocation

    def next_action(self, observation: str) -> tuple[str, dict]:
        action = self.script[self._idx]
        self._idx += 1
        self._last_invocation = LLMInvocation(
            request_messages=[{"role": "user", "content": observation}],
            response_raw={"tool": action[0]},
            model_api_id="fake-model",
            input_tokens=10,
            output_tokens=5,
            latency_ms=42,
        )
        return action

    def observe(self, message: str) -> None:
        pass


def test_log_entries_have_board_state():
    """Each tool-call log entry should include board_before and board_after."""
    p0 = FakePlayerWithInvocation([("forfeit", {})], "Alice")
    p1 = FakePlayerWithInvocation([], "Bob")

    runner = GameRunner(players=[p0, p1], max_turns=10)
    result = runner.play()

    assert len(result.log) >= 1
    entry = result.log[0]
    assert "board_before" in entry
    assert "board_after" in entry
    assert entry["board_before"] == [0, 0]


def test_log_entries_have_invocation():
    """Each tool-call log entry should include the LLM invocation snapshot."""
    p0 = FakePlayerWithInvocation([("forfeit", {})], "Alice")
    p1 = FakePlayerWithInvocation([], "Bob")

    runner = GameRunner(players=[p0, p1], max_turns=10)
    result = runner.play()

    entry = result.log[0]
    assert "invocation" in entry
    inv = entry["invocation"]
    assert isinstance(inv, LLMInvocation)
    assert inv.model_api_id == "fake-model"
    assert inv.input_tokens == 10


def test_log_entries_have_result_fields():
    """Each tool-call log entry should include validation result fields."""
    p0 = FakePlayerWithInvocation([("forfeit", {})], "Alice")
    p1 = FakePlayerWithInvocation([], "Bob")

    runner = GameRunner(players=[p0, p1], max_turns=10)
    result = runner.play()

    entry = result.log[0]
    assert "result_ok" in entry
    assert "result_message" in entry
    assert entry["result_ok"] is True


def test_log_entries_have_turn_number():
    """Each log entry should include the turn_number."""
    p0 = FakePlayerWithInvocation([("forfeit", {})], "Alice")
    p1 = FakePlayerWithInvocation([], "Bob")

    runner = GameRunner(players=[p0, p1], max_turns=10)
    result = runner.play()

    entry = result.log[0]
    assert "turn_number" in entry
    assert entry["turn_number"] == 1
