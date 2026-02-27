"""RED — tests that GameRunner captures board state and invocation data via observer."""

from chutes_bench.board import BoardState
from chutes_bench.game import GameRunner, ListObserver, LogEntry
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
    """Each log entry should include board_before and board_after."""
    p0 = FakePlayerWithInvocation([("forfeit", {})], "Alice")
    p1 = FakePlayerWithInvocation([], "Bob")

    observer = ListObserver()
    runner = GameRunner(players=[p0, p1], max_turns=10, observer=observer)
    runner.play()

    assert len(observer.entries) >= 1
    entry = observer.entries[0]
    assert isinstance(entry, LogEntry)
    assert entry.board_before == [0, 0]


def test_log_entries_have_invocation():
    """Each log entry should include the LLM invocation snapshot."""
    p0 = FakePlayerWithInvocation([("forfeit", {})], "Alice")
    p1 = FakePlayerWithInvocation([], "Bob")

    observer = ListObserver()
    runner = GameRunner(players=[p0, p1], max_turns=10, observer=observer)
    runner.play()

    entry = observer.entries[0]
    assert entry.invocation is not None
    assert isinstance(entry.invocation, LLMInvocation)
    assert entry.invocation.model_api_id == "fake-model"
    assert entry.invocation.input_tokens == 10


def test_log_entries_have_result_fields():
    """Each log entry should include validation result fields."""
    p0 = FakePlayerWithInvocation([("forfeit", {})], "Alice")
    p1 = FakePlayerWithInvocation([], "Bob")

    observer = ListObserver()
    runner = GameRunner(players=[p0, p1], max_turns=10, observer=observer)
    runner.play()

    entry = observer.entries[0]
    assert entry.result_ok is True
    assert entry.result_message != ""


def test_log_entries_have_turn_number():
    """Each log entry should include the turn_number."""
    p0 = FakePlayerWithInvocation([("forfeit", {})], "Alice")
    p1 = FakePlayerWithInvocation([], "Bob")

    observer = ListObserver()
    runner = GameRunner(players=[p0, p1], max_turns=10, observer=observer)
    runner.play()

    entry = observer.entries[0]
    assert entry.turn_number == 1
