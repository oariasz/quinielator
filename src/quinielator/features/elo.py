"""Rating Elo cronológico y sin información futura."""

from __future__ import annotations

import math
from collections import defaultdict


class EloRatingCalculator:
    """Mantiene ratings que solo cambian después de observar un partido."""

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 30.0,
        home_advantage: float = 0.0,
    ) -> None:
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self._ratings: defaultdict[str, float] = defaultdict(lambda: self.initial_rating)

    def rating(self, team_code: str) -> float:
        """Rating conocido en este instante."""

        return float(self._ratings[team_code])

    def expected_home(self, home_team: str, away_team: str) -> float:
        """Probabilidad Elo de victoria local sin modelar explícitamente el empate."""

        difference = self.rating(away_team) - (self.rating(home_team) + self.home_advantage)
        return float(1.0 / (1.0 + 10.0 ** (difference / 400.0)))

    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
    ) -> None:
        """Actualiza ambos equipos tras conocer el marcador."""

        home_rating = self.rating(home_team)
        away_rating = self.rating(away_team)
        expected = self.expected_home(home_team, away_team)
        if home_goals > away_goals:
            actual = 1.0
        elif home_goals < away_goals:
            actual = 0.0
        else:
            actual = 0.5
        margin = abs(home_goals - away_goals)
        margin_multiplier = 1.0 if margin <= 1 else math.log1p(margin)
        change = self.k_factor * margin_multiplier * (actual - expected)
        self._ratings[home_team] = home_rating + change
        self._ratings[away_team] = away_rating - change
