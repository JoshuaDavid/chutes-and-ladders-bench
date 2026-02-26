"""Elo rating calculation from pairwise game outcomes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Outcome:
    """Result of a single game."""

    player_a: str
    player_b: str
    winner: str | None  # None = draw


def compute_elo(
    outcomes: list[Outcome],
    initial: float = 1500.0,
    k: float = 32.0,
) -> dict[str, float]:
    """Compute Elo ratings from a list of outcomes.

    Uses the standard Elo formula:
      E_a = 1 / (1 + 10^((R_b - R_a) / 400))
      R_a' = R_a + K * (S_a - E_a)

    where S_a is 1 for win, 0.5 for draw, 0 for loss.
    """
    ratings: dict[str, float] = {}

    for outcome in outcomes:
        for p in (outcome.player_a, outcome.player_b):
            if p not in ratings:
                ratings[p] = initial

        ra = ratings[outcome.player_a]
        rb = ratings[outcome.player_b]

        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea

        if outcome.winner is None:
            sa, sb = 0.5, 0.5
        elif outcome.winner == outcome.player_a:
            sa, sb = 1.0, 0.0
        else:
            sa, sb = 0.0, 1.0

        ratings[outcome.player_a] = ra + k * (sa - ea)
        ratings[outcome.player_b] = rb + k * (sb - eb)

    return ratings
