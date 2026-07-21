"""Baseline interpretable basado en media de goles y diferencia Elo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quinielator.models.base import BatchPrediction, PredictionModel, score_probabilities


class BaselineModel(PredictionModel):
    """Referencia mínima que todo modelo más complejo debería superar."""

    name = "baseline"

    def __init__(self, max_score: int = 8, seed: int = 42) -> None:
        super().__init__(max_score, seed)
        self.home_average = 1.4
        self.away_average = 1.1

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> BaselineModel:
        del features
        self.home_average = max(float(targets["home_goals"].mean()), 0.1)
        self.away_average = max(float(targets["away_goals"].mean()), 0.1)
        return self

    def predict(self, features: pd.DataFrame) -> BatchPrediction:
        elo_difference = features["elo_difference"].fillna(0.0).to_numpy(dtype=float)
        adjustment = np.clip(elo_difference / 800.0, -0.6, 0.6)
        home_expected = self.home_average * np.exp(adjustment)
        away_expected = self.away_average * np.exp(-adjustment)
        return score_probabilities(home_expected, away_expected, self.max_score)
