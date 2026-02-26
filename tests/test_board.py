"""RED — tests for chutes_bench.board (written before implementation)."""

from chutes_bench.board import (
    CHUTES_LADDERS,
    BoardState,
    is_ladder,
    is_chute,
    chute_or_ladder_dest,
    apply_spin,
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
    assert b.current_player == 0


def test_first_spin_places_pawn():
    """First spin from square 0 → land on that square number."""
    b = BoardState()
    result = apply_spin(b, player=0, spin=3)
    assert result.new_position == 3
    assert not result.won


def test_first_spin_hits_ladder():
    """Spin 1 on first turn → land on 1 → ladder to 38."""
    b = BoardState()
    result = apply_spin(b, player=0, spin=1)
    assert result.new_position == 1  # pre-chute/ladder position
    assert result.chute_or_ladder_dest == 38


def test_normal_move():
    b = BoardState(positions=[10, 0])
    result = apply_spin(b, player=0, spin=4)
    assert result.new_position == 14
    assert not result.won


def test_exact_landing_on_100_wins():
    b = BoardState(positions=[96, 0])
    result = apply_spin(b, player=0, spin=4)
    assert result.new_position == 100
    assert result.won


def test_overshoot_100_stays_put():
    """If spin would take you past 100, you don't move."""
    b = BoardState(positions=[96, 0])
    result = apply_spin(b, player=0, spin=5)
    assert result.new_position == 96
    assert result.bounced


def test_landing_on_chute():
    b = BoardState(positions=[10, 0])
    result = apply_spin(b, player=0, spin=6)  # 10 + 6 = 16 → chute to 6
    assert result.new_position == 16
    assert result.chute_or_ladder_dest == 6


def test_landing_on_ladder_28():
    b = BoardState(positions=[25, 0])
    result = apply_spin(b, player=0, spin=3)  # 25 + 3 = 28 → ladder to 84
    assert result.new_position == 28
    assert result.chute_or_ladder_dest == 84


def test_landing_on_80_ladder_wins():
    """Square 80 has a ladder to 100 — that should count as a win."""
    b = BoardState(positions=[77, 0])
    result = apply_spin(b, player=0, spin=3)  # 77 + 3 = 80 → ladder to 100
    assert result.new_position == 80
    assert result.chute_or_ladder_dest == 100
    assert result.won
