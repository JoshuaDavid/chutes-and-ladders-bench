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
]


# ── Turn-phase tracker ───────────────────────────────────────────────

@dataclass
class TurnPhase:
    """Tracks what a player has done so far this turn."""

    has_spun: bool = False
    spin_value: int | None = None
    has_moved: bool = False
    moved_to: int | None = None
    needs_chute_or_ladder: str | None = None  # "chute" | "ladder" | None
    chute_or_ladder_done: bool = False
    draw_offered_to_me: bool = False


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
        return ActionResult(ok=True, spin_value=value, message=f"You spun a {value}.")

    # ── move_pawn_to_square ──
    if tool_name == "move_pawn_to_square":
        if not phase.has_spun:
            return ActionResult(ok=False, illegal=True, message="You must spin first.")
        if phase.has_moved:
            return ActionResult(ok=False, illegal=True, message="Already moved this turn.")

        target_square = args.get("square")
        if target_square is None:
            return ActionResult(ok=False, illegal=True, message="Missing 'square' argument.")

        spin_result = apply_spin(board, player, phase.spin_value)

        # Bounce case: player must pass their current position
        if spin_result.bounced:
            if target_square != board.positions[player]:
                return ActionResult(
                    ok=False, illegal=True,
                    message=f"Spin overshoots 100. You must stay on {board.positions[player]}.",
                )
            phase.has_moved = True
            return ActionResult(ok=True, bounced=True, message="Spin overshoots 100. You stay put.")

        if target_square != spin_result.new_position:
            return ActionResult(
                ok=False, illegal=True,
                message=f"Wrong square. Expected {spin_result.new_position}, got {target_square}.",
            )

        phase.has_moved = True
        phase.moved_to = target_square

        # Check chute/ladder
        if is_ladder(target_square):
            phase.needs_chute_or_ladder = "ladder"
            return ActionResult(
                ok=True, requires_ladder=True,
                message=f"Moved to {target_square}. There's a ladder here!",
            )
        if is_chute(target_square):
            phase.needs_chute_or_ladder = "chute"
            return ActionResult(
                ok=True, requires_chute=True,
                message=f"Moved to {target_square}. There's a chute here!",
            )

        won = target_square == 100
        return ActionResult(ok=True, won=won, message=f"Moved to {target_square}.")

    # ── ascend_ladder_to_square ──
    if tool_name == "ascend_ladder_to_square":
        if phase.needs_chute_or_ladder != "ladder":
            return ActionResult(ok=False, illegal=True, message="No ladder to ascend.")
        dest = args.get("square")
        expected = CHUTES_LADDERS.get(phase.moved_to, -1)
        if dest != expected:
            return ActionResult(
                ok=False, illegal=True,
                message=f"Wrong ladder destination. Expected {expected}, got {dest}.",
            )
        phase.needs_chute_or_ladder = None
        phase.chute_or_ladder_done = True
        won = dest == 100
        return ActionResult(ok=True, won=won, message=f"Climbed ladder to {dest}!")

    # ── descend_chute_to_square ──
    if tool_name == "descend_chute_to_square":
        if phase.needs_chute_or_ladder != "chute":
            return ActionResult(ok=False, illegal=True, message="No chute to descend.")
        dest = args.get("square")
        expected = CHUTES_LADDERS.get(phase.moved_to, -1)
        if dest != expected:
            return ActionResult(
                ok=False, illegal=True,
                message=f"Wrong chute destination. Expected {expected}, got {dest}.",
            )
        phase.needs_chute_or_ladder = None
        phase.chute_or_ladder_done = True
        return ActionResult(ok=True, message=f"Slid down chute to {dest}.")

    # ── end_turn ──
    if tool_name == "end_turn":
        if not phase.has_moved:
            return ActionResult(ok=False, illegal=True, message="You must move before ending your turn.")
        if phase.needs_chute_or_ladder is not None:
            return ActionResult(
                ok=False, illegal=True,
                message=f"You must take the {phase.needs_chute_or_ladder} before ending your turn.",
            )
        return ActionResult(ok=True, turn_over=True, message="Turn over.")

    return ActionResult(ok=False, illegal=True, message=f"Unknown tool: {tool_name}")
