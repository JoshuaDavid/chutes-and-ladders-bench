"""CLI entry point: python -m chutes_bench {run,elo,chart}."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

from chutes_bench.elo import Outcome, compute_elo
from chutes_bench.chart import make_elo_chart
from chutes_bench.game import GameRunner
from chutes_bench.players import MODELS, ModelSpec


RESULTS_DIR = Path("results")


# ── run ──────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> None:
    """Run trials for every ordered pairing of models."""
    RESULTS_DIR.mkdir(exist_ok=True)

    models = MODELS
    if args.models:
        allowed = set(args.models)
        models = [m for m in MODELS if m.display_name in allowed or m.id in allowed]

    pairings = list(itertools.permutations(models, 2))
    total = len(pairings) * args.trials
    print(f"Running {total} games ({len(pairings)} pairings × {args.trials} trials)")

    outcomes: list[dict] = []

    for i, (spec_a, spec_b) in enumerate(pairings):
        for trial in range(args.trials):
            label = f"[{i * args.trials + trial + 1}/{total}] {spec_a.display_name} vs {spec_b.display_name}"
            print(f"{label} ... ", end="", flush=True)

            player_a = spec_a.make_player()
            player_b = spec_b.make_player()

            runner = GameRunner(
                players=[player_a, player_b],
                max_turns=args.max_turns,
            )
            result = runner.play()

            if result.winner == 0:
                winner_name = spec_a.display_name
            elif result.winner == 1:
                winner_name = spec_b.display_name
            else:
                winner_name = None

            outcome = {
                "player_a": spec_a.display_name,
                "player_b": spec_b.display_name,
                "winner": winner_name,
                "reason": result.reason,
                "turns": result.turns,
            }
            outcomes.append(outcome)
            print(f"{result.reason} → {winner_name or 'draw'}")

    # Save results
    out_path = RESULTS_DIR / "outcomes.json"
    # Append to existing results if any
    existing: list[dict] = []
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    existing.extend(outcomes)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {out_path} ({len(existing)} total games)")


# ── elo ──────────────────────────────────────────────────────────────

def cmd_elo(args: argparse.Namespace) -> None:
    """Compute and print Elo ratings from saved results."""
    path = RESULTS_DIR / "outcomes.json"
    if not path.exists():
        print(f"No results found at {path}. Run some games first.", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        raw = json.load(f)

    outcomes = [
        Outcome(
            player_a=r["player_a"],
            player_b=r["player_b"],
            winner=r["winner"],
        )
        for r in raw
    ]

    ratings = compute_elo(outcomes)
    print("\nElo Ratings")
    print("=" * 40)
    for name, rating in sorted(ratings.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:30s} {rating:7.1f}")


# ── chart ────────────────────────────────────────────────────────────

def cmd_chart(args: argparse.Namespace) -> None:
    """Generate leaderboard chart from saved results."""
    path = RESULTS_DIR / "outcomes.json"
    if not path.exists():
        print(f"No results found at {path}. Run some games first.", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        raw = json.load(f)

    outcomes = [
        Outcome(player_a=r["player_a"], player_b=r["player_b"], winner=r["winner"])
        for r in raw
    ]
    ratings = compute_elo(outcomes)

    out = args.output or "elo_leaderboard.png"
    make_elo_chart(ratings, output_path=out)
    print(f"Chart saved to {out}")


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="chutes_bench",
        description="Chutes & Ladders LLM Elo Benchmark",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run game trials")
    p_run.add_argument("--trials", type=int, default=3, help="Trials per pairing (default 3)")
    p_run.add_argument("--max-turns", type=int, default=200, help="Max turns per game")
    p_run.add_argument("--models", nargs="*", help="Subset of model names to include")

    p_elo = sub.add_parser("elo", help="Compute Elo ratings")

    p_chart = sub.add_parser("chart", help="Generate leaderboard chart")
    p_chart.add_argument("--output", "-o", help="Output PNG path")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "elo":
        cmd_elo(args)
    elif args.command == "chart":
        cmd_chart(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
