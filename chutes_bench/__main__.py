"""CLI entry point: python -m chutes_bench {run,elo,chart}."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chutes_bench.elo import compute_elo
from chutes_bench.chart import make_elo_chart
from chutes_bench.game import GameRunner, GameResult
from chutes_bench.invocation import LLMInvocation
from chutes_bench.persistence import ResultsDB
from chutes_bench.players import MODELS, SYSTEM_PROMPT


RESULTS_DIR = Path("results")
DB_PATH = RESULTS_DIR / "benchmark.db"


def _open_db() -> ResultsDB:
    RESULTS_DIR.mkdir(exist_ok=True)
    return ResultsDB(DB_PATH)


def _persist_game_log(
    db: ResultsDB,
    game_id: int,
    result: GameResult,
) -> None:
    """Write the enriched game log to the turns, llm_invocations, and tool_calls tables."""
    # Group log entries by turn_number to build turn summaries
    turns_seen: dict[int, list[dict]] = {}
    for entry in result.log:
        tn = entry["turn_number"]
        turns_seen.setdefault(tn, []).append(entry)

    invocation_seq: dict[int, int] = {}  # turn_number → next sequence counter

    for turn_number, entries in sorted(turns_seen.items()):
        player_idx = entries[0]["player"]
        start_pos = entries[0]["board_before"][player_idx]
        end_pos = entries[-1]["board_after"][player_idx]
        spin_value = None
        outcome = "normal"

        for e in entries:
            if e["tool"] == "spin_spinner" and e["result_ok"]:
                # Extract spin value from result message: "You spun a N."
                msg = e["result_message"]
                try:
                    spin_value = int(msg.split()[-1].rstrip("."))
                except (ValueError, IndexError):
                    pass
            if e.get("is_winning_move"):
                outcome = "win"
            elif e.get("is_illegal"):
                outcome = "illegal_move"
            elif e.get("is_forfeit"):
                outcome = "forfeit"

        # Check for draw / bounce
        if result.reason == "draw" and turn_number == result.turns:
            outcome = "draw"

        db.record_turn(
            game_id=game_id,
            turn_number=turn_number,
            player_idx=player_idx,
            start_position=start_pos,
            end_position=end_pos,
            spin_value=spin_value,
            outcome=outcome,
            actions_count=len(entries),
        )

        for entry in entries:
            seq = invocation_seq.get(turn_number, 0)
            invocation_seq[turn_number] = seq + 1

            inv: LLMInvocation | None = entry.get("invocation")
            if inv is not None:
                inv_id = db.record_llm_invocation(
                    game_id=game_id,
                    turn_number=turn_number,
                    player_idx=entry["player"],
                    sequence_in_turn=seq,
                    model_api_id=inv.model_api_id,
                    request_messages=inv.request_messages,
                    response_raw=inv.response_raw,
                    input_tokens=inv.input_tokens,
                    output_tokens=inv.output_tokens,
                    latency_ms=inv.latency_ms,
                )
            else:
                # Player doesn't expose invocations (e.g. FakePlayer in tests)
                inv_id = db.record_llm_invocation(
                    game_id=game_id,
                    turn_number=turn_number,
                    player_idx=entry["player"],
                    sequence_in_turn=seq,
                    model_api_id="unknown",
                    request_messages=[],
                    response_raw={},
                )

            db.record_tool_call(
                invocation_id=inv_id,
                game_id=game_id,
                turn_number=turn_number,
                player_idx=entry["player"],
                tool_name=entry["tool"],
                tool_args=entry["args"],
                result_ok=entry["result_ok"],
                result_message=entry["result_message"],
                board_before=entry["board_before"],
                board_after=entry["board_after"],
                is_winning_move=entry.get("is_winning_move", False),
                is_illegal=entry.get("is_illegal", False),
                is_forfeit=entry.get("is_forfeit", False),
                is_turn_over=entry.get("is_turn_over", False),
            )


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

        runner = GameRunner(
            players=[player_a, player_b],
            max_turns=args.max_turns,
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
        _persist_game_log(db, game_id, result)

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
