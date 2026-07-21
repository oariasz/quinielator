import pytest

from quinielator.features import EloRatingCalculator


def test_elo_winner_gains_and_sum_is_conserved() -> None:
    elo = EloRatingCalculator(initial_rating=1500.0, k_factor=30.0)
    elo.update("AAA", "BBB", 2, 0)
    assert elo.rating("AAA") > 1500
    assert elo.rating("BBB") < 1500
    assert elo.rating("AAA") + elo.rating("BBB") == pytest.approx(3000.0)


def test_equal_ratings_have_half_expectation() -> None:
    elo = EloRatingCalculator()
    assert elo.expected_home("AAA", "BBB") == pytest.approx(0.5)
