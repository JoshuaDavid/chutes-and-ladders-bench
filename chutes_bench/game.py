"""Game runner — orchestrates a two-player Chutes & Ladders game."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from chutes_bench.board import BoardState, CHUTES_LADDERS
from chutes_bench.tools import TurnPhase, validate_action, ActionResult


# ── Player interface ─────────────────────────────────────────────────

class Player(ABC):
    """Interface that every player (LLM or fake) must implement."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def next_action(self, observation: str) -> tuple[str, dict]:
        """Return (tool_name, args) for the next action."""
        ...

    @abstractmethod
    def observe(self, message: str) -> None:
        """Receive a message (result of action, opponent chat, etc.)."""
        ...


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

    def play(self) -> GameResult:
        turns = 0

        while turns < self.max_turns:
            for player_idx in range(2):
                result = self._play_turn(player_idx)
                turns += 1

                if result is not None:
                    result.turns = turns
                    result.log = self.log
                    return result

                if turns >= self.max_turns:
                    return GameResult(
                        winner=None, reason="max_turns",
                        turns=turns, log=self.log,
                    )

        return GameResult(winner=None, reason="max_turns", turns=turns, log=self.log)

    def _play_turn(self, player_idx: int) -> GameResult | None:
        """Run one player's full turn. Returns GameResult if game ends."""
        player = self.players[player_idx]
        opponent_idx = 1 - player_idx
        phase = TurnPhase()

        # If a draw was offered to this player, let them know
        if self.draw_offered_by == opponent_idx:
            phase.draw_offered_to_me = True

        observation = self._make_observation(player_idx)

        for _ in range(MAX_ACTIONS_PER_TURN):
            tool_name, args = player.next_action(observation)

            self.log.append({
                "player": player_idx,
                "tool": tool_name,
                "args": args,
            })

            result = validate_action(
                self.board, player_idx, tool_name, args, phase,
            )

            if not result.ok:
                # Illegal move → automatic loss
                self.log.append({"event": "illegal_move", "player": player_idx, "msg": result.message})
                return GameResult(winner=opponent_idx, reason="illegal_move")

            # Handle special results
            if result.forfeit:
                return GameResult(winner=opponent_idx, reason="forfeit")

            if result.draw:
                return GameResult(winner=None, reason="draw")

            if tool_name == "offer_draw":
                self.draw_offered_by = player_idx

            if result.won:
                # Commit position before returning
                final_pos = phase.moved_to or self.board.positions[player_idx]
                cl_dest = CHUTES_LADDERS.get(final_pos)
                if cl_dest == 100:
                    self.board.positions[player_idx] = 100
                else:
                    self.board.positions[player_idx] = final_pos
                return GameResult(winner=player_idx, reason="win")

            if result.turn_over:
                # Commit final position
                if phase.chute_or_ladder_done and phase.moved_to is not None:
                    self.board.positions[player_idx] = CHUTES_LADDERS[phase.moved_to]
                elif phase.moved_to is not None:
                    self.board.positions[player_idx] = phase.moved_to
                break

            # Feed result back as next observation
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
