"""RED — tests for chutes_bench.elo."""

from chutes_bench.elo import compute_elo, Outcome


def test_two_players_one_win_each():
    """Equal results → ratings stay near 1500."""
    outcomes = [
        Outcome(player_a="A", player_b="B", winner="A"),
        Outcome(player_a="A", player_b="B", winner="B"),
    ]
    ratings = compute_elo(outcomes)
    # Not exactly symmetric because game order matters in Elo,
    # but should be close (within ~5 points).
    assert abs(ratings["A"] - ratings["B"]) < 5


def test_dominant_player_rated_higher():
    outcomes = [
        Outcome(player_a="A", player_b="B", winner="A"),
        Outcome(player_a="A", player_b="B", winner="A"),
        Outcome(player_a="A", player_b="B", winner="A"),
        Outcome(player_a="A", player_b="B", winner="A"),
    ]
    ratings = compute_elo(outcomes)
    assert ratings["A"] > ratings["B"]


def test_draw_keeps_ratings_close():
    outcomes = [
        Outcome(player_a="A", player_b="B", winner=None),  # draw
    ]
    ratings = compute_elo(outcomes)
    assert abs(ratings["A"] - ratings["B"]) < 1


def test_three_players():
    outcomes = [
        Outcome(player_a="A", player_b="B", winner="A"),
        Outcome(player_a="B", player_b="C", winner="B"),
        Outcome(player_a="A", player_b="C", winner="A"),
    ]
    ratings = compute_elo(outcomes)
    assert ratings["A"] > ratings["B"] > ratings["C"]


def test_initial_rating_is_1500():
    outcomes = [
        Outcome(player_a="X", player_b="Y", winner="X"),
    ]
    ratings = compute_elo(outcomes, initial=1500)
    assert ratings["X"] > 1500
    assert ratings["Y"] < 1500
