"""CLI entry point: python -m chutes_bench {run,elo,chart}."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chutes_bench.elo import compute_elo
from chutes_bench.chart import make_elo_chart
from chutes_bench.game import GameRunner
from chutes_bench.persistence import ResultsDB
from chutes_bench.players import MODELS


RESULTS_DIR = Path("results")
DB_PATH = RESULTS_DIR / "benchmark.db"


def _open_db() -> ResultsDB:
    RESULTS_DIR.mkdir(exist_ok=True)
    return ResultsDB(DB_PATH)


# ── run ──────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> None:
    """Run trials for every ordered pairing, resuming where we left off."""
    db = _open_db()

    models = MODELS
    if args.models:
        allowed = set(args.models)
        models = [m for m in MODELS if m.display_name in allowed or m.id in allowed]

    model_names = [m.display_name for m in models]
    name_to_spec = {m.display_name: m for m in models}

    db.ensure_pairings(model_names, trials=args.trials)
    pending = db.pending_pairings()

    if not pending:
        print("All pairings already completed. Nothing to do.")
        return

    print(f"{len(pending)} games remaining")

    for i, pairing in enumerate(pending):
        spec_a = name_to_spec.get(pairing.player_a)
        spec_b = name_to_spec.get(pairing.player_b)
        if spec_a is None or spec_b is None:
            continue

        label = f"[{i + 1}/{len(pending)}] {pairing.player_a} vs {pairing.player_b} (trial {pairing.trial})"
        print(f"{label} ... ", end="", flush=True)

        player_a = spec_a.make_player()
        player_b = spec_b.make_player()

        runner = GameRunner(
            players=[player_a, player_b],
            max_turns=args.max_turns,
        )
        result = runner.play()

        if result.winner == 0:
            winner_name = pairing.player_a
        elif result.winner == 1:
            winner_name = pairing.player_b
        else:
            winner_name = None

        db.record_game(
            player_a=pairing.player_a,
            player_b=pairing.player_b,
            winner=winner_name,
            reason=result.reason,
            turns=result.turns,
            pairing_id=pairing.id,
        )
        print(f"{result.reason} → {winner_name or 'draw'}")

    remaining = db.pending_pairings()
    print(f"\nDone. {len(remaining)} games still pending.")
    db.close()


# ── elo ──────────────────────────────────────────────────────────────

def cmd_elo(args: argparse.Namespace) -> None:
    """Compute and print Elo ratings from the database."""
    if not DB_PATH.exists():
        print(f"No database found at {DB_PATH}. Run some games first.", file=sys.stderr)
        sys.exit(1)

    db = _open_db()
    outcomes = db.list_outcomes()
    db.close()

    if not outcomes:
        print("No completed games yet.", file=sys.stderr)
        sys.exit(1)

    ratings = compute_elo(outcomes)
    print("\nElo Ratings")
    print("=" * 40)
    for name, rating in sorted(ratings.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:30s} {rating:7.1f}")


# ── chart ────────────────────────────────────────────────────────────

def cmd_chart(args: argparse.Namespace) -> None:
    """Generate leaderboard chart from the database."""
    if not DB_PATH.exists():
        print(f"No database found at {DB_PATH}. Run some games first.", file=sys.stderr)
        sys.exit(1)

    db = _open_db()
    outcomes = db.list_outcomes()
    db.close()

    if not outcomes:
        print("No completed games yet.", file=sys.stderr)
        sys.exit(1)

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

    p_run = sub.add_parser("run", help="Run game trials (resumes automatically)")
    p_run.add_argument("--trials", type=int, default=3, help="Trials per pairing (default 3)")
    p_run.add_argument("--max-turns", type=int, default=200, help="Max turns per game")
    p_run.add_argument("--models", nargs="*", help="Subset of model names to include")

    sub.add_parser("elo", help="Compute Elo ratings")

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
