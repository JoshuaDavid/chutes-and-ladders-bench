"""SQLite persistence for benchmark results.

Supports pause/resume: pairings are created up front, then completed
one at a time. Rerunning picks up where you left off.
"""

from __future__ import annotations

import itertools
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from chutes_bench.elo import Outcome


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
        """)
        self._conn.commit()

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

    def close(self) -> None:
        self._conn.close()
