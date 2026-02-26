"""RED — tests for SQLite persistence layer."""

import sqlite3
import tempfile
from pathlib import Path

from chutes_bench.persistence import ResultsDB


def test_create_db():
    """Creating a DB initializes the schema."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = ResultsDB(db_path)
        # Tables should exist
        conn = sqlite3.connect(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "games" in tables
        assert "pairings" in tables


def test_record_and_list_outcomes():
    with tempfile.TemporaryDirectory() as tmp:
        db = ResultsDB(Path(tmp) / "test.db")
        db.record_game(
            player_a="Alice", player_b="Bob",
            winner="Alice", reason="win", turns=10,
        )
        db.record_game(
            player_a="Alice", player_b="Bob",
            winner="Bob", reason="win", turns=15,
        )
        outcomes = db.list_outcomes()
        assert len(outcomes) == 2
        assert outcomes[0].player_a == "Alice"
        assert outcomes[1].winner == "Bob"


def test_pending_pairings():
    """Mark pairings as pending, complete them, check remaining."""
    with tempfile.TemporaryDirectory() as tmp:
        db = ResultsDB(Path(tmp) / "test.db")
        models = ["A", "B", "C"]
        db.ensure_pairings(models, trials=2)

        pending = db.pending_pairings()
        # 3 models → 6 ordered pairings × 2 trials = 12
        assert len(pending) == 12

        # Complete one
        p = pending[0]
        db.record_game(
            player_a=p.player_a, player_b=p.player_b,
            winner=p.player_a, reason="win", turns=5,
            pairing_id=p.id,
        )

        assert len(db.pending_pairings()) == 11


def test_resume_is_idempotent():
    """Calling ensure_pairings again doesn't duplicate."""
    with tempfile.TemporaryDirectory() as tmp:
        db = ResultsDB(Path(tmp) / "test.db")
        db.ensure_pairings(["A", "B"], trials=3)
        count1 = len(db.pending_pairings())

        db.ensure_pairings(["A", "B"], trials=3)
        count2 = len(db.pending_pairings())
        assert count1 == count2


def test_list_outcomes_returns_elo_outcomes():
    """list_outcomes returns Outcome objects compatible with compute_elo."""
    from chutes_bench.elo import compute_elo

    with tempfile.TemporaryDirectory() as tmp:
        db = ResultsDB(Path(tmp) / "test.db")
        db.record_game("X", "Y", winner="X", reason="win", turns=5)
        db.record_game("X", "Y", winner="Y", reason="win", turns=8)

        outcomes = db.list_outcomes()
        ratings = compute_elo(outcomes)
        assert "X" in ratings and "Y" in ratings
