"""Game runner — orchestrates a two-player Chutes & Ladders game."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from chutes_bench.board import BoardState
from chutes_bench.invocation import LLMInvocation
from chutes_bench.tools import TurnPhase, validate_action


# ── Player interface ─────────────────────────────────────────────────

@runtime_checkable
class Player(Protocol):
    """Structural interface — any object with these methods works."""

    @property
    def name(self) -> str: ...

    @property
    def last_invocation(self) -> LLMInvocation | None: ...

    def next_action(self, observation: str) -> tuple[str, dict]: ...

    def observe(self, message: str) -> None: ...


# ── Structured types ────────────────────────────────────────────────

@dataclass
class LogEntry:
    """Record of a single tool-call action during a game."""

    turn_number: int
    player: int
    tool: str
    args: dict
    board_before: list[int]
    board_after: list[int]
    result_ok: bool
    result_message: str
    is_winning_move: bool = False
    is_illegal: bool = False
    is_forfeit: bool = False
    is_turn_over: bool = False
    spin_value: int | None = None
    invocation: LLMInvocation | None = None


@dataclass
class GameState:
    """Mutable state that evolves during a game."""

    board: BoardState = field(default_factory=BoardState)
    turn_number: int = 0
    draw_offered_by: int | None = None


@dataclass
class GameResult:
    winner: int | None  # 0 or 1, or None for draw
    reason: str  # "win" | "forfeit" | "illegal_move" | "draw" | "max_turns"
    turns: int = 0


# ── Observer ────────────────────────────────────────────────────────

class GameObserver(Protocol):
    """Receives structured events as a game is played."""

    def on_action(self, entry: LogEntry) -> None: ...


@dataclass
class ListObserver:
    """Default observer — collects entries into a list."""

    entries: list[LogEntry] = field(default_factory=list)

    def on_action(self, entry: LogEntry) -> None:
        self.entries.append(entry)


# ── Runner ───────────────────────────────────────────────────────────

MAX_ACTIONS_PER_TURN = 20  # safety valve against infinite tool calls


class GameRunner:
    """Play one full game between two players."""

    def __init__(
        self,
        players: list[Player],
        state: GameState | None = None,
        max_turns: int = 200,
        observer: GameObserver | None = None,
    ):
        assert len(players) == 2
        self.players = players
        self.state = state or GameState()
        self.max_turns = max_turns
        self.observer = observer or ListObserver()

    def play(self) -> GameResult:
        self.state.turn_number = 0

        while self.state.turn_number < self.max_turns:
            for player_idx in range(2):
                result = self._play_turn(player_idx)
                self.state.turn_number += 1

                if result is not None:
                    result.turns = self.state.turn_number
                    return result

                if self.state.turn_number >= self.max_turns:
                    return GameResult(
                        winner=None, reason="max_turns",
                        turns=self.state.turn_number,
                    )

        return GameResult(
            winner=None, reason="max_turns",
            turns=self.state.turn_number,
        )

    def _play_turn(self, player_idx: int) -> GameResult | None:
        """Run one player's full turn. Returns GameResult if game ends."""
        player = self.players[player_idx]
        opponent_idx = 1 - player_idx
        board = self.state.board
        phase = TurnPhase(start_position=board.positions[player_idx])

        if self.state.draw_offered_by == opponent_idx:
            phase.draw_offered_to_me = True

        observation = self._make_observation(player_idx)
        turn_number = self.state.turn_number + 1  # 1-indexed for entries

        for _ in range(MAX_ACTIONS_PER_TURN):
            tool_name, args = player.next_action(observation)

            board_before = list(board.positions)

            result = validate_action(
                board, player_idx, tool_name, args, phase,
            )

            # Start building the log entry — board_after filled in below
            entry = LogEntry(
                turn_number=turn_number,
                player=player_idx,
                tool=tool_name,
                args=args,
                board_before=board_before,
                board_after=board_before,  # default: no change
                result_ok=result.ok,
                result_message=result.message,
                is_winning_move=result.won,
                is_illegal=not result.ok,
                is_forfeit=result.forfeit,
                is_turn_over=result.turn_over,
                spin_value=result.spin_value,
                invocation=player.last_invocation,
            )

            if not result.ok:
                self.observer.on_action(entry)
                return GameResult(winner=opponent_idx, reason="illegal_move")

            if result.forfeit:
                self.observer.on_action(entry)
                return GameResult(winner=opponent_idx, reason="forfeit")

            if result.draw:
                self.observer.on_action(entry)
                return GameResult(winner=None, reason="draw")

            if tool_name == "offer_draw":
                self.state.draw_offered_by = player_idx

            # Commit board position on turn-ending actions (won or end_turn)
            if result.won or result.turn_over:
                if phase.current_position is not None:
                    board.positions[player_idx] = phase.current_position
                entry.board_after = list(board.positions)
                self.observer.on_action(entry)
                if result.won:
                    return GameResult(winner=player_idx, reason="win")
                break  # turn_over

            # Mid-turn action (spin, move, etc.) — board not committed yet
            self.observer.on_action(entry)

            observation = result.message
            player.observe(result.message)

        return None

    def _make_observation(self, player_idx: int) -> str:
        opponent_idx = 1 - player_idx
        board = self.state.board
        my_pos = board.positions[player_idx]
        opp_pos = board.positions[opponent_idx]
        msg = f"Your turn. You are on square {my_pos}. Opponent is on square {opp_pos}."
        if my_pos == 0:
            msg = f"Your turn. You are not yet on the board. Opponent is on square {opp_pos}."
        if self.state.draw_offered_by == opponent_idx:
            msg += " Your opponent has offered a draw."
        return msg
