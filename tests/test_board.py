"""RED — tests for chutes_bench.board (written before implementation)."""

from chutes_bench.board import (
    CHUTES_LADDERS,
    BoardState,
    is_ladder,
    is_chute,
    chute_or_ladder_dest,
)


# ── constants ────────────────────────────────────────────────────────

def test_chutes_ladders_has_9_ladders():
    ladders = {sq: dest for sq, dest in CHUTES_LADDERS.items() if dest > sq}
    assert len(ladders) == 9


def test_chutes_ladders_has_10_chutes():
    chutes = {sq: dest for sq, dest in CHUTES_LADDERS.items() if dest < sq}
    assert len(chutes) == 10


def test_all_squares_in_range():
    for sq, dest in CHUTES_LADDERS.items():
        assert 1 <= sq <= 100
        assert 1 <= dest <= 100


def test_is_ladder():
    assert is_ladder(1) is True      # 1 → 38
    assert is_ladder(28) is True     # 28 → 84
    assert is_ladder(50) is False
    assert is_ladder(16) is False    # chute, not ladder


def test_is_chute():
    assert is_chute(16) is True      # 16 → 6
    assert is_chute(98) is True      # 98 → 78
    assert is_chute(50) is False
    assert is_chute(1) is False      # ladder, not chute


def test_chute_or_ladder_dest():
    assert chute_or_ladder_dest(1) == 38
    assert chute_or_ladder_dest(98) == 78
    assert chute_or_ladder_dest(50) is None


# ── BoardState ───────────────────────────────────────────────────────

def test_initial_state():
    b = BoardState()
    assert b.positions == [0, 0]
