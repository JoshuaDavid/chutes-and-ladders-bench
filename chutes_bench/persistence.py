"""SQLite persistence for benchmark results.

Supports pause/resume: pairings are created up front, then completed
one at a time. Rerunning picks up where you left off.
"""

from __future__ import annotations

import itertools
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chutes_bench.elo import Outcome

if TYPE_CHECKING:
    from chutes_bench.game import LogEntry


@dataclass
class PendingPairing:
    id: int
    player_a: str
    player_b: str
    trial: int


class ResultsDB:
    """Thin wrapper around a SQLite database for game outcomes."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS pairings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                player_a    TEXT NOT NULL,
                player_b    TEXT NOT NULL,
                trial       INTEGER NOT NULL,
                completed   INTEGER NOT NULL DEFAULT 0,
                UNIQUE(player_a, player_b, trial)
            );
            CREATE TABLE IF NOT EXISTS games (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                pairing_id  INTEGER REFERENCES pairings(id),
                player_a    TEXT NOT NULL,
                player_b    TEXT NOT NULL,
                winner      TEXT,
                reason      TEXT NOT NULL,
                turns       INTEGER NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS models (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                api_id          TEXT NOT NULL UNIQUE,
                display_name    TEXT NOT NULL UNIQUE,
                provider        TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id         INTEGER NOT NULL REFERENCES games(id),
                turn_number     INTEGER NOT NULL,
                player_idx      INTEGER NOT NULL,
                start_position  INTEGER NOT NULL,
                end_position    INTEGER NOT NULL,
                spin_value      INTEGER,
                outcome         TEXT NOT NULL,
                actions_count   INTEGER NOT NULL,
                UNIQUE(game_id, turn_number)
            );
            CREATE TABLE IF NOT EXISTS llm_invocations (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id          INTEGER NOT NULL REFERENCES games(id),
                turn_number      INTEGER NOT NULL,
                player_idx       INTEGER NOT NULL,
                sequence_in_turn INTEGER NOT NULL,
                model_api_id     TEXT NOT NULL,
                request_messages TEXT NOT NULL,
                response_raw     TEXT NOT NULL,
                input_tokens     INTEGER,
                output_tokens    INTEGER,
                latency_ms       INTEGER,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(game_id, turn_number, sequence_in_turn)
            );
            CREATE TABLE IF NOT EXISTS tool_calls (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                invocation_id    INTEGER NOT NULL REFERENCES llm_invocations(id),
                game_id          INTEGER NOT NULL REFERENCES games(id),
                turn_number      INTEGER NOT NULL,
                player_idx       INTEGER NOT NULL,
                tool_name        TEXT NOT NULL,
                tool_args        TEXT NOT NULL,
                result_ok        INTEGER NOT NULL,
                result_message   TEXT NOT NULL,
                board_before     TEXT NOT NULL,
                board_after      TEXT NOT NULL,
                is_winning_move  INTEGER NOT NULL DEFAULT 0,
                is_illegal       INTEGER NOT NULL DEFAULT 0,
                is_forfeit       INTEGER NOT NULL DEFAULT 0,
                is_turn_over     INTEGER NOT NULL DEFAULT 0
            );
        """)
        self._conn.commit()

    # ── Legacy API (unchanged) ──────────────────────────────────────

    def ensure_pairings(self, model_names: list[str], trials: int) -> None:
        """Insert all pairings × trials, skipping duplicates."""
        for a, b in itertools.permutations(model_names, 2):
            for trial in range(trials):
                self._conn.execute(
                    "INSERT OR IGNORE INTO pairings (player_a, player_b, trial) VALUES (?, ?, ?)",
                    (a, b, trial),
                )
        self._conn.commit()

    def pending_pairings(self) -> list[PendingPairing]:
        """Return all pairings that haven't been completed yet."""
        rows = self._conn.execute(
            "SELECT id, player_a, player_b, trial FROM pairings WHERE completed = 0 ORDER BY id"
        ).fetchall()
        return [PendingPairing(id=r[0], player_a=r[1], player_b=r[2], trial=r[3]) for r in rows]

    def record_game(
        self,
        player_a: str,
        player_b: str,
        winner: str | None,
        reason: str,
        turns: int,
        pairing_id: int | None = None,
    ) -> None:
        """Record a completed game and mark its pairing done."""
        self._conn.execute(
            "INSERT INTO games (pairing_id, player_a, player_b, winner, reason, turns) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (pairing_id, player_a, player_b, winner, reason, turns),
        )
        if pairing_id is not None:
            self._conn.execute(
                "UPDATE pairings SET completed = 1 WHERE id = ?", (pairing_id,)
            )
        self._conn.commit()

    def list_outcomes(self) -> list[Outcome]:
        """Return all completed games as Elo-compatible Outcome objects."""
        rows = self._conn.execute(
            "SELECT player_a, player_b, winner FROM games ORDER BY id"
        ).fetchall()
        return [Outcome(player_a=r[0], player_b=r[1], winner=r[2]) for r in rows]

    # ── Detailed logging API ────────────────────────────────────────

    def ensure_model(self, api_id: str, display_name: str, provider: str) -> int:
        """Insert a model if it doesn't exist. Returns the model row id."""
        self._conn.execute(
            "INSERT OR IGNORE INTO models (api_id, display_name, provider) VALUES (?, ?, ?)",
            (api_id, display_name, provider),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id FROM models WHERE api_id = ?", (api_id,)
        ).fetchone()
        return row[0]

    def _model_display_name(self, model_id: int) -> str | None:
        row = self._conn.execute(
            "SELECT display_name FROM models WHERE id = ?", (model_id,)
        ).fetchone()
        return row[0] if row else None

    def record_game_detailed(
        self,
        player_a_model_id: int,
        player_b_model_id: int,
        winner_idx: int | None,
        reason: str,
        total_turns: int,
        max_turns_limit: int,
        system_prompt: str,
        pairing_id: int | None = None,
    ) -> int:
        """Record a game with full metadata. Also writes to legacy games table."""
        name_a = self._model_display_name(player_a_model_id) or "?"
        name_b = self._model_display_name(player_b_model_id) or "?"

        if winner_idx == 0:
            winner_name = name_a
        elif winner_idx == 1:
            winner_name = name_b
        else:
            winner_name = None

        # Write to legacy games table so Elo/chart still work
        cur = self._conn.execute(
            "INSERT INTO games (pairing_id, player_a, player_b, winner, reason, turns) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (pairing_id, name_a, name_b, winner_name, reason, total_turns),
        )
        game_id = cur.lastrowid

        if pairing_id is not None:
            self._conn.execute(
                "UPDATE pairings SET completed = 1 WHERE id = ?", (pairing_id,)
            )
        self._conn.commit()
        return game_id  # type: ignore[return-value]

    def record_turn(
        self,
        game_id: int,
        turn_number: int,
        player_idx: int,
        start_position: int,
        end_position: int,
        spin_value: int | None,
        outcome: str,
        actions_count: int,
    ) -> int:
        """Record a single player-turn summary."""
        cur = self._conn.execute(
            "INSERT INTO turns (game_id, turn_number, player_idx, start_position, "
            "end_position, spin_value, outcome, actions_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (game_id, turn_number, player_idx, start_position, end_position,
             spin_value, outcome, actions_count),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def record_llm_invocation(
        self,
        game_id: int,
        turn_number: int,
        player_idx: int,
        sequence_in_turn: int,
        model_api_id: str,
        request_messages: list[dict],
        response_raw: dict,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: int | None = None,
    ) -> int:
        """Record a single LLM API call."""
        cur = self._conn.execute(
            "INSERT INTO llm_invocations (game_id, turn_number, player_idx, "
            "sequence_in_turn, model_api_id, request_messages, response_raw, "
            "input_tokens, output_tokens, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (game_id, turn_number, player_idx, sequence_in_turn, model_api_id,
             json.dumps(request_messages), json.dumps(response_raw),
             input_tokens, output_tokens, latency_ms),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def record_tool_call(
        self,
        invocation_id: int,
        game_id: int,
        turn_number: int,
        player_idx: int,
        tool_name: str,
        tool_args: dict,
        result_ok: bool,
        result_message: str,
        board_before: list[int],
        board_after: list[int],
        is_winning_move: bool = False,
        is_illegal: bool = False,
        is_forfeit: bool = False,
        is_turn_over: bool = False,
    ) -> int:
        """Record a single tool call with its validation result."""
        cur = self._conn.execute(
            "INSERT INTO tool_calls (invocation_id, game_id, turn_number, player_idx, "
            "tool_name, tool_args, result_ok, result_message, board_before, board_after, "
            "is_winning_move, is_illegal, is_forfeit, is_turn_over) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (invocation_id, game_id, turn_number, player_idx,
             tool_name, json.dumps(tool_args),
             int(result_ok), result_message,
             json.dumps(board_before), json.dumps(board_after),
             int(is_winning_move), int(is_illegal),
             int(is_forfeit), int(is_turn_over)),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def close(self) -> None:
        self._conn.close()


# ── Game log persistence ─────────────────────────────────────────────

def persist_game_log(
    db: ResultsDB,
    game_id: int,
    entries: list[LogEntry],
    reason: str,
    total_turns: int,
) -> None:
    """Write structured game events to the turns, llm_invocations, and tool_calls tables."""
    from chutes_bench.invocation import LLMInvocation

    # Group entries by turn_number to build turn summaries
    turns_seen: dict[int, list[LogEntry]] = {}
    for entry in entries:
        turns_seen.setdefault(entry.turn_number, []).append(entry)

    invocation_seq: dict[int, int] = {}  # turn_number → next sequence counter

    for turn_number, turn_entries in sorted(turns_seen.items()):
        first = turn_entries[0]
        last = turn_entries[-1]
        player_idx = first.player
        start_pos = first.board_before[player_idx]
        end_pos = last.board_after[player_idx]
        spin_value = None
        outcome = "normal"

        for e in turn_entries:
            if e.spin_value is not None:
                spin_value = e.spin_value
            if e.is_winning_move:
                outcome = "win"
            elif e.is_illegal:
                outcome = "illegal_move"
            elif e.is_forfeit:
                outcome = "forfeit"

        if reason == "draw" and turn_number == total_turns:
            outcome = "draw"

        db.record_turn(
            game_id=game_id,
            turn_number=turn_number,
            player_idx=player_idx,
            start_position=start_pos,
            end_position=end_pos,
            spin_value=spin_value,
            outcome=outcome,
            actions_count=len(turn_entries),
        )

        for entry in turn_entries:
            seq = invocation_seq.get(turn_number, 0)
            invocation_seq[turn_number] = seq + 1

            inv: LLMInvocation | None = entry.invocation
            if inv is not None:
                inv_id = db.record_llm_invocation(
                    game_id=game_id,
                    turn_number=turn_number,
                    player_idx=entry.player,
                    sequence_in_turn=seq,
                    model_api_id=inv.model_api_id,
                    request_messages=inv.request_messages,
                    response_raw=inv.response_raw,
                    input_tokens=inv.input_tokens,
                    output_tokens=inv.output_tokens,
                    latency_ms=inv.latency_ms,
                )
            else:
                inv_id = db.record_llm_invocation(
                    game_id=game_id,
                    turn_number=turn_number,
                    player_idx=entry.player,
                    sequence_in_turn=seq,
                    model_api_id="unknown",
                    request_messages=[],
                    response_raw={},
                )

            db.record_tool_call(
                invocation_id=inv_id,
                game_id=game_id,
                turn_number=turn_number,
                player_idx=entry.player,
                tool_name=entry.tool,
                tool_args=entry.args,
                result_ok=entry.result_ok,
                result_message=entry.result_message,
                board_before=entry.board_before,
                board_after=entry.board_after,
                is_winning_move=entry.is_winning_move,
                is_illegal=entry.is_illegal,
                is_forfeit=entry.is_forfeit,
                is_turn_over=entry.is_turn_over,
            )
