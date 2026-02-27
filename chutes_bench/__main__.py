"""CLI entry point: python -m chutes_bench {run,elo,chart}."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chutes_bench.elo import compute_elo
from chutes_bench.chart import make_elo_chart
from chutes_bench.export import generate_all, upload_to_b2
from chutes_bench.game import GameRunner, ListObserver
from chutes_bench.persistence import ResultsDB, persist_game_log
from chutes_bench.players import MODELS, SYSTEM_PROMPT


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

    # Pre-register all models
    model_ids = {}
    for m in models:
        model_ids[m.display_name] = db.ensure_model(m.id, m.display_name, m.provider)

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

        observer = ListObserver()
        runner = GameRunner(
            players=[player_a, player_b],
            max_turns=args.max_turns,
            observer=observer,
        )
        result = runner.play()

        game_id = db.record_game_detailed(
            player_a_model_id=model_ids[pairing.player_a],
            player_b_model_id=model_ids[pairing.player_b],
            winner_idx=result.winner,
            reason=result.reason,
            total_turns=result.turns,
            max_turns_limit=args.max_turns,
            system_prompt=SYSTEM_PROMPT,
            pairing_id=pairing.id,
        )
        persist_game_log(
            db, game_id, observer.entries,
            reason=result.reason, total_turns=result.turns,
        )

        if result.winner == 0:
            winner_name = pairing.player_a
        elif result.winner == 1:
            winner_name = pairing.player_b
        else:
            winner_name = None
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


# ── export ────────────────────────────────────────────────────────────

def cmd_export(args: argparse.Namespace) -> None:
    """Export game data to JSON, optionally uploading to Backblaze B2."""
    if not DB_PATH.exists():
        print(f"No database found at {DB_PATH}. Run some games first.", file=sys.stderr)
        sys.exit(1)

    import os
    output_dir = Path(args.output_dir)
    generated = generate_all(DB_PATH, output_dir)
    print(f"Generated {len(generated)} file(s) in {output_dir}/")

    if args.upload:
        bucket = os.environ.get("B2_BUCKET_NAME", "")
        endpoint = os.environ.get("B2_ENDPOINT_URL", "")
        key_id = os.environ.get("B2_KEY_ID", "")
        app_key = os.environ.get("B2_APP_KEY", "")

        missing = []
        if not bucket:
            missing.append("B2_BUCKET_NAME")
        if not endpoint:
            missing.append("B2_ENDPOINT_URL")
        if not key_id:
            missing.append("B2_KEY_ID")
        if not app_key:
            missing.append("B2_APP_KEY")
        if missing:
            print(f"Missing env vars for B2 upload: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)

        prefix = args.prefix
        files: dict[str, bytes] = {}
        for path in generated:
            rel = path.relative_to(output_dir)
            key = f"{prefix}/{rel}" if prefix else str(rel)
            files[key] = path.read_bytes()

        upload_to_b2(files, bucket, endpoint, key_id, app_key)
        print(f"Uploaded {len(files)} file(s) to B2 bucket '{bucket}'.")


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

    p_export = sub.add_parser("export", help="Export game data to JSON (+ optional B2 upload)")
    p_export.add_argument("--output-dir", "-o", default="export_data", help="Local output directory (default: export_data)")
    p_export.add_argument("--upload", action="store_true", help="Upload to Backblaze B2 (needs B2_* env vars)")
    p_export.add_argument("--prefix", default="data", help="Object key prefix in B2 (default: data)")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "elo":
        cmd_elo(args)
    elif args.command == "chart":
        cmd_chart(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
