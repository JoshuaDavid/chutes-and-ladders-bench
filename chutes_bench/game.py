"""Game runner — orchestrates a two-player Chutes & Ladders game."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from chutes_bench.board import BoardState, CHUTES_LADDERS
from chutes_bench.invocation import LLMInvocation
from chutes_bench.tools import TurnPhase, validate_action, ActionResult


# ── Player interface ─────────────────────────────────────────────────

@runtime_checkable
class Player(Protocol):
    """Structural interface — any object with these methods works."""

    @property
    def name(self) -> str: ...

    def next_action(self, observation: str) -> tuple[str, dict]: ...

    def observe(self, message: str) -> None: ...


# ── Game result ──────────────────────────────────────────────────────

@dataclass
class GameResult:
    winner: int | None  # 0 or 1, or None for draw
    reason: str  # "win" | "forfeit" | "illegal_move" | "draw" | "max_turns"
    turns: int = 0
    log: list[dict] = field(default_factory=list)


# ── Runner ───────────────────────────────────────────────────────────

MAX_ACTIONS_PER_TURN = 20  # safety valve against infinite tool calls


class GameRunner:
    """Play one full game between two players."""

    def __init__(
        self,
        players: list[Player],
        max_turns: int = 200,
        board: BoardState | None = None,
    ):
        assert len(players) == 2
        self.players = players
        self.max_turns = max_turns
        self.board = board or BoardState()
        self.log: list[dict] = []
        self.draw_offered_by: int | None = None
        self._turn_number = 0

    def play(self) -> GameResult:
        self._turn_number = 0

        while self._turn_number < self.max_turns:
            for player_idx in range(2):
                result = self._play_turn(player_idx)
                self._turn_number += 1

                if result is not None:
                    result.turns = self._turn_number
                    result.log = self.log
                    return result

                if self._turn_number >= self.max_turns:
                    return GameResult(
                        winner=None, reason="max_turns",
                        turns=self._turn_number, log=self.log,
                    )

        return GameResult(winner=None, reason="max_turns", turns=self._turn_number, log=self.log)

    def _play_turn(self, player_idx: int) -> GameResult | None:
        """Run one player's full turn. Returns GameResult if game ends."""
        player = self.players[player_idx]
        opponent_idx = 1 - player_idx
        phase = TurnPhase(start_position=self.board.positions[player_idx])

        if self.draw_offered_by == opponent_idx:
            phase.draw_offered_to_me = True

        observation = self._make_observation(player_idx)
        turn_number = self._turn_number + 1  # 1-indexed for log entries

        for _ in range(MAX_ACTIONS_PER_TURN):
            tool_name, args = player.next_action(observation)

            # Capture board state before validation
            board_before = list(self.board.positions)

            result = validate_action(
                self.board, player_idx, tool_name, args, phase,
            )

            # Capture invocation if the player exposes it
            invocation = getattr(player, "last_invocation", None)

            # Build enriched log entry
            log_entry: dict = {
                "turn_number": turn_number,
                "player": player_idx,
                "tool": tool_name,
                "args": args,
                "board_before": board_before,
                "result_ok": result.ok,
                "result_message": result.message,
                "is_winning_move": result.won,
                "is_illegal": not result.ok,
                "is_forfeit": result.forfeit,
                "is_turn_over": result.turn_over,
            }
            if result.spin_value is not None:
                log_entry["spin_value"] = result.spin_value
            if invocation is not None:
                log_entry["invocation"] = invocation

            if not result.ok:
                # Board doesn't change on illegal move
                log_entry["board_after"] = board_before
                self.log.append(log_entry)
                return GameResult(winner=opponent_idx, reason="illegal_move")

            if result.forfeit:
                log_entry["board_after"] = board_before
                self.log.append(log_entry)
                return GameResult(winner=opponent_idx, reason="forfeit")

            if result.draw:
                log_entry["board_after"] = board_before
                self.log.append(log_entry)
                return GameResult(winner=None, reason="draw")

            if tool_name == "offer_draw":
                self.draw_offered_by = player_idx

            if result.won:
                self.board.positions[player_idx] = phase.current_position or 100
                log_entry["board_after"] = list(self.board.positions)
                self.log.append(log_entry)
                return GameResult(winner=player_idx, reason="win")

            if result.turn_over:
                if phase.current_position is not None:
                    self.board.positions[player_idx] = phase.current_position
                log_entry["board_after"] = list(self.board.positions)
                self.log.append(log_entry)
                break

            # Mid-turn action (spin, move, etc.) — board hasn't been committed yet
            log_entry["board_after"] = board_before
            self.log.append(log_entry)

            observation = result.message
            player.observe(result.message)

        return None

    def _make_observation(self, player_idx: int) -> str:
        opponent_idx = 1 - player_idx
        my_pos = self.board.positions[player_idx]
        opp_pos = self.board.positions[opponent_idx]
        msg = f"Your turn. You are on square {my_pos}. Opponent is on square {opp_pos}."
        if my_pos == 0:
            msg = f"Your turn. You are not yet on the board. Opponent is on square {opp_pos}."
        if self.draw_offered_by == opponent_idx:
            msg += " Your opponent has offered a draw."
        return msg
