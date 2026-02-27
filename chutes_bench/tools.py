"""Tool schemas (OpenAI function-calling format) and action validation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from chutes_bench.board import (
    BoardState,
    CHUTES_LADDERS,
    apply_spin,
    is_chute,
    is_ladder,
)

# ── OpenAI-style tool schemas ────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "spin_spinner",
            "description": "Spin the spinner to get a value 1–6. Must be called once at the start of your turn.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_pawn_to_square",
            "description": "Move your pawn to the target square (current position + spin value). If the spin would take you past 100 you stay put — pass your current square.",
            "parameters": {
                "type": "object",
                "properties": {
                    "square": {
                        "type": "integer",
                        "description": "The square to move to.",
                    }
                },
                "required": ["square"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ascend_ladder_to_square",
            "description": "Take a ladder from your current square to its destination. Only valid when you just landed on a ladder base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "square": {
                        "type": "integer",
                        "description": "The destination square at the top of the ladder.",
                    }
                },
                "required": ["square"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "descend_chute_to_square",
            "description": "Slide down a chute from your current square to its destination. Only valid when you just landed on a chute top.",
            "parameters": {
                "type": "object",
                "properties": {
                    "square": {
                        "type": "integer",
                        "description": "The destination square at the bottom of the chute.",
                    }
                },
                "required": ["square"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_turn",
            "description": "Signal that your turn is over.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a chat message to your opponent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send.",
                    }
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forfeit",
            "description": "Forfeit the game (automatic loss).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "offer_draw",
            "description": "Offer a draw to your opponent.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accept_draw",
            "description": "Accept an offered draw.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan",
            "description": "Think step-by-step about your next actions. This tool has no side effects — use it to reason before acting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your internal reasoning or plan for this turn.",
                    }
                },
                "required": ["thought"],
            },
        },
    },
]


# ── Turn-phase tracker ───────────────────────────────────────────────

@dataclass
class TurnPhase:
    """Tracks what a player has done so far this turn.

    Supports multi-step movement: a player may call move_pawn_to_square
    multiple times to walk through intermediate squares before arriving
    at the final resting position.
    """

    has_spun: bool = False
    spin_value: int | None = None
    # Populated by the game runner (or test) before validation starts
    start_position: int = 0
    # Tracks where the pawn is right now during multi-step movement
    current_position: int | None = None  # None = hasn't moved yet
    # Set once the pawn reaches the final valid resting position
    reached_final: bool = False
    draw_offered_to_me: bool = False

    # Legacy compat — kept for game.py commit logic
    @property
    def has_moved(self) -> bool:
        return self.current_position is not None

    @property
    def moved_to(self) -> int | None:
        return self.current_position

    @property
    def chute_or_ladder_done(self) -> bool:
        """True if we've passed through a chute/ladder to the final dest."""
        if not self.reached_final or self.current_position is None:
            return False
        landing = self._landing_square()
        if landing is None:
            return False
        cl_dest = CHUTES_LADDERS.get(landing)
        return cl_dest is not None and self.current_position == cl_dest

    def _landing_square(self) -> int | None:
        if self.spin_value is None:
            return None
        target = self.start_position + self.spin_value
        if target > 100:
            return None  # bounce
        return target

    def _final_resting_square(self) -> int | None:
        """Where the pawn must end up this turn."""
        landing = self._landing_square()
        if landing is None:
            # Bounce → stay put
            return self.start_position if self.spin_value is not None else None
        cl_dest = CHUTES_LADDERS.get(landing)
        if cl_dest is not None:
            return cl_dest
        return landing


# ── Action result ────────────────────────────────────────────────────

@dataclass
class ActionResult:
    ok: bool = True
    message: str = ""
    illegal: bool = False
    turn_over: bool = False
    forfeit: bool = False
    won: bool = False
    draw: bool = False
    spin_value: int | None = None
    requires_ladder: bool = False
    requires_chute: bool = False
    bounced: bool = False


# ── Validation logic ─────────────────────────────────────────────────

def validate_action(
    board: BoardState,
    player: int,
    tool_name: str,
    args: dict,
    phase: TurnPhase,
) -> ActionResult:
    """Validate and (partially) execute a tool call.

    Returns an ActionResult. Mutates *phase* to track turn progress.
    Does NOT mutate *board* — the game runner commits state.
    """
    if tool_name == "plan":
        return ActionResult(ok=True, message=f"Plan noted.")

    if tool_name == "send_message":
        return ActionResult(ok=True, message=f"Message sent: {args.get('message', '')}")

    if tool_name == "forfeit":
        return ActionResult(ok=True, forfeit=True, message="Player forfeits.")

    if tool_name == "offer_draw":
        return ActionResult(ok=True, message="Draw offered.")

    if tool_name == "accept_draw":
        if phase.draw_offered_to_me:
            return ActionResult(ok=True, draw=True, message="Draw accepted.")
        return ActionResult(ok=False, illegal=True, message="No draw has been offered.")

    # ── spin_spinner ──
    if tool_name == "spin_spinner":
        if phase.has_spun:
            return ActionResult(ok=False, illegal=True, message="Already spun this turn.")
        value = random.randint(1, 6)
        phase.has_spun = True
        phase.spin_value = value
        phase.start_position = board.positions[player]
        return ActionResult(ok=True, spin_value=value, message=f"You spun a {value}.")

    # ── move_pawn_to_square ──
    if tool_name == "move_pawn_to_square":
        return _validate_move(board, player, args, phase)

    # ── ascend_ladder_to_square ──
    if tool_name == "ascend_ladder_to_square":
        return _validate_ladder(args, phase)

    # ── descend_chute_to_square ──
    if tool_name == "descend_chute_to_square":
        return _validate_chute(args, phase)

    # ── end_turn ──
    if tool_name == "end_turn":
        return _validate_end_turn(phase)

    return ActionResult(ok=False, illegal=True, message=f"Unknown tool: {tool_name}")


def _validate_move(
    board: BoardState, player: int, args: dict, phase: TurnPhase,
) -> ActionResult:
    if not phase.has_spun:
        return ActionResult(ok=False, illegal=True, message="You must spin first.")
    if phase.reached_final:
        return ActionResult(ok=False, illegal=True, message="Already at final position.")

    target_square = args.get("square")
    if target_square is None:
        return ActionResult(ok=False, illegal=True, message="Missing 'square' argument.")

    landing = phase._landing_square()
    final_resting = phase._final_resting_square()
    cur = phase.current_position if phase.current_position is not None else phase.start_position

    # ── Bounce case ──
    if landing is None:
        if target_square != phase.start_position:
            return ActionResult(
                ok=False, illegal=True,
                message=f"Spin overshoots 100. You must stay on {phase.start_position}.",
            )
        phase.current_position = phase.start_position
        phase.reached_final = True
        return ActionResult(ok=True, bounced=True, message="Spin overshoots 100. You stay put.")

    # ── Direct jump to final resting position (e.g. move(14) when ladder 4→14) ──
    if target_square == final_resting and final_resting != landing:
        phase.current_position = final_resting
        phase.reached_final = True
        won = final_resting == 100
        return ActionResult(ok=True, won=won, message=f"Moved to {final_resting}.")

    # ── Intermediate or landing square ──
    # Must be forward from current position
    if target_square <= cur:
        return ActionResult(
            ok=False, illegal=True,
            message=f"Can't move backward. Current position is {cur}, tried {target_square}.",
        )
    # Must not go past the landing square
    if target_square > landing:
        return ActionResult(
            ok=False, illegal=True,
            message=f"Square {target_square} is past your landing square {landing}.",
        )

    phase.current_position = target_square

    # Did we reach the landing square?
    if target_square == landing:
        if is_ladder(landing):
            return ActionResult(
                ok=True, requires_ladder=True,
                message=f"Moved to {landing}. There's a ladder here!",
            )
        if is_chute(landing):
            return ActionResult(
                ok=True, requires_chute=True,
                message=f"Moved to {landing}. There's a chute here!",
            )
        # Plain landing square, no chute/ladder
        phase.reached_final = True
        won = landing == 100
        return ActionResult(ok=True, won=won, message=f"Moved to {landing}.")

    # Intermediate square
    return ActionResult(ok=True, message=f"Moved to {target_square}.")


def _validate_ladder(args: dict, phase: TurnPhase) -> ActionResult:
    landing = phase._landing_square()
    if landing is None or phase.current_position != landing or not is_ladder(landing):
        return ActionResult(ok=False, illegal=True, message="No ladder to ascend.")

    dest = args.get("square")
    expected = CHUTES_LADDERS[landing]
    if dest != expected:
        return ActionResult(
            ok=False, illegal=True,
            message=f"Wrong ladder destination. Expected {expected}, got {dest}.",
        )
    phase.current_position = dest
    phase.reached_final = True
    won = dest == 100
    return ActionResult(ok=True, won=won, message=f"Climbed ladder to {dest}!")


def _validate_chute(args: dict, phase: TurnPhase) -> ActionResult:
    landing = phase._landing_square()
    if landing is None or phase.current_position != landing or not is_chute(landing):
        return ActionResult(ok=False, illegal=True, message="No chute to descend.")

    dest = args.get("square")
    expected = CHUTES_LADDERS[landing]
    if dest != expected:
        return ActionResult(
            ok=False, illegal=True,
            message=f"Wrong chute destination. Expected {expected}, got {dest}.",
        )
    phase.current_position = dest
    phase.reached_final = True
    return ActionResult(ok=True, message=f"Slid down chute to {dest}.")


def _validate_end_turn(phase: TurnPhase) -> ActionResult:
    if not phase.has_moved:
        return ActionResult(ok=False, illegal=True, message="You must move before ending your turn.")
    if not phase.reached_final:
        return ActionResult(
            ok=False, illegal=True,
            message=f"You haven't reached your final square yet. Currently on {phase.current_position}.",
        )
    return ActionResult(ok=True, turn_over=True, message="Turn over.")
