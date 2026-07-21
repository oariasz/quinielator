"""Modelos de scikit-learn para goles y resultado 1X2."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quinielator.models.base import BatchPrediction, PredictionModel, score_probabilities


def _poisson_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.7, max_iter=1000)),
        ]
    )


class SklearnPoissonModel(PredictionModel):
    """Dos regresiones Poisson para obtener una distribución de marcadores."""

    name = "poisson"

    def __init__(self, max_score: int = 8, seed: int = 42) -> None:
        super().__init__(max_score, seed)
        self.home_model: Pipeline = _poisson_pipeline()
        self.away_model: Pipeline = _poisson_pipeline()

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> SklearnPoissonModel:
        self.home_model.fit(features, targets["home_goals"])
        self.away_model.fit(features, targets["away_goals"])
        return self

    def predict(self, features: pd.DataFrame) -> BatchPrediction:
        home = np.clip(self.home_model.predict(features), 0.05, 5.0)
        away = np.clip(self.away_model.predict(features), 0.05, 5.0)
        return score_probabilities(home, away, self.max_score)


class SklearnOutcomeModel(PredictionModel):
    """Regresión logística multinomial calibrada por su propia función softmax."""

    name = "outcome"

    def __init__(self, max_score: int = 8, seed: int = 42) -> None:
        super().__init__(max_score, seed)
        self.pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        C=0.35,
                        random_state=seed,
                    ),
                ),
            ]
        )
        self.home_average = 1.4
        self.away_average = 1.1

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> SklearnOutcomeModel:
        self.pipeline.fit(features, targets["outcome"])
        self.home_average = max(float(targets["home_goals"].mean()), 0.1)
        self.away_average = max(float(targets["away_goals"].mean()), 0.1)
        return self

    def predict(self, features: pd.DataFrame) -> BatchPrediction:
        raw = self.pipeline.predict_proba(features)
        classifier = self.pipeline.named_steps["model"]
        classes = list(classifier.classes_)
        probability_home = raw[:, classes.index("H")]
        probability_draw = raw[:, classes.index("D")]
        probability_away = raw[:, classes.index("A")]
        strength = np.clip(probability_home - probability_away, -0.7, 0.7)
        home = self.home_average * np.exp(strength)
        away = self.away_average * np.exp(-strength)
        poisson_output = score_probabilities(home, away, self.max_score)
        return BatchPrediction(
            expected_home_goals=poisson_output.expected_home_goals,
            expected_away_goals=poisson_output.expected_away_goals,
            predicted_home_goals=poisson_output.predicted_home_goals,
            predicted_away_goals=poisson_output.predicted_away_goals,
            probability_home=probability_home,
            probability_draw=probability_draw,
            probability_away=probability_away,
        )
