"""Board state and movement rules for Chutes & Ladders."""

from __future__ import annotations

from dataclasses import dataclass, field

# fmt: off
CHUTES_LADDERS: dict[int, int] = {
    # Ladders (go UP)
     1: 38,   4: 14,   9: 31,  21: 42,  28: 84,
    36: 44,  51: 67,  71: 91,  80: 100,
    # Chutes (go DOWN)
    16:  6,  47: 26,  49: 11,  56: 53,  62: 19,
    64: 60,  87: 24,  93: 73,  95: 75,  98: 78,
}
# fmt: on


def is_ladder(square: int) -> bool:
    dest = CHUTES_LADDERS.get(square)
    return dest is not None and dest > square


def is_chute(square: int) -> bool:
    dest = CHUTES_LADDERS.get(square)
    return dest is not None and dest < square


def chute_or_ladder_dest(square: int) -> int | None:
    return CHUTES_LADDERS.get(square)


@dataclass
class BoardState:
    """Mutable state for a two-player game."""

    positions: list[int] = field(default_factory=lambda: [0, 0])
    current_player: int = 0


@dataclass
class SpinResult:
    """What happened after a spin."""

    new_position: int
    chute_or_ladder_dest: int | None = None
    won: bool = False
    bounced: bool = False


def apply_spin(board: BoardState, player: int, spin: int) -> SpinResult:
    """Compute the result of *player* spinning *spin*.

    Does NOT mutate *board* — the caller decides whether to commit.
    """
    old = board.positions[player]
    target = old + spin

    # Overshoot → stay put
    if target > 100:
        return SpinResult(new_position=old, bounced=True)

    # Check for chute/ladder
    cl_dest = chute_or_ladder_dest(target)
    won = target == 100 or cl_dest == 100

    return SpinResult(
        new_position=target,
        chute_or_ladder_dest=cl_dest,
        won=won,
    )
