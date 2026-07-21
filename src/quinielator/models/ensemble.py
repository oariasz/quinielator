"""Ensemble temporal entre Poisson y Keras."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from quinielator.models.base import BatchPrediction, PredictionModel, score_probabilities
from quinielator.models.keras_model import KerasMatchModel
from quinielator.models.sklearn_models import SklearnPoissonModel


class EnsembleModel(PredictionModel):
    """Selecciona el peso con el último bloque temporal del entrenamiento."""

    name = "ensemble"

    def __init__(
        self,
        max_score: int = 8,
        seed: int = 42,
        keras_epochs: int = 80,
        keras_batch_size: int = 32,
        keras_patience: int = 10,
    ) -> None:
        super().__init__(max_score, seed)
        self.keras_epochs = keras_epochs
        self.keras_batch_size = keras_batch_size
        self.keras_patience = keras_patience
        self.poisson = SklearnPoissonModel(max_score, seed)
        self.keras = KerasMatchModel(
            max_score,
            seed,
            keras_epochs,
            keras_batch_size,
            keras_patience,
        )
        self.poisson_weight = 1.0
        self.validation_log_loss: float | None = None

    def _new_keras(self) -> KerasMatchModel:
        return KerasMatchModel(
            self.max_score,
            self.seed,
            self.keras_epochs,
            self.keras_batch_size,
            self.keras_patience,
        )

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> EnsembleModel:
        if not KerasMatchModel.available() or len(features) < 160:
            self.poisson.fit(features, targets)
            self.poisson_weight = 1.0
            return self

        split = int(len(features) * 0.80)
        train_x, validation_x = features.iloc[:split], features.iloc[split:]
        train_y, validation_y = targets.iloc[:split], targets.iloc[split:]
        poisson_validation = SklearnPoissonModel(self.max_score, self.seed).fit(train_x, train_y)
        keras_validation = self._new_keras().fit(train_x, train_y)
        poisson_output = poisson_validation.predict(validation_x)
        keras_output = keras_validation.predict(validation_x)
        actual = validation_y["outcome"].to_numpy()
        best_loss = float("inf")
        best_weight = 1.0
        for weight in np.linspace(0.0, 1.0, 5):
            probabilities = np.column_stack(
                [
                    weight * poisson_output.probability_away
                    + (1.0 - weight) * keras_output.probability_away,
                    weight * poisson_output.probability_draw
                    + (1.0 - weight) * keras_output.probability_draw,
                    weight * poisson_output.probability_home
                    + (1.0 - weight) * keras_output.probability_home,
                ]
            )
            loss = float(log_loss(actual, probabilities, labels=["A", "D", "H"]))
            if loss < best_loss:
                best_loss = loss
                best_weight = float(weight)
        self.poisson_weight = best_weight
        self.validation_log_loss = best_loss
        self.poisson.fit(features, targets)
        self.keras.fit(features, targets)
        return self

    def predict(self, features: pd.DataFrame) -> BatchPrediction:
        poisson_output = self.poisson.predict(features)
        if self.poisson_weight >= 1.0 or self.keras.model is None:
            return poisson_output
        keras_output = self.keras.predict(features)
        weight = self.poisson_weight
        home_goals = (
            weight * poisson_output.expected_home_goals
            + (1.0 - weight) * keras_output.expected_home_goals
        )
        away_goals = (
            weight * poisson_output.expected_away_goals
            + (1.0 - weight) * keras_output.expected_away_goals
        )
        score_output = score_probabilities(home_goals, away_goals, self.max_score)
        return BatchPrediction(
            expected_home_goals=home_goals,
            expected_away_goals=away_goals,
            predicted_home_goals=score_output.predicted_home_goals,
            predicted_away_goals=score_output.predicted_away_goals,
            probability_home=(
                weight * poisson_output.probability_home
                + (1.0 - weight) * keras_output.probability_home
            ),
            probability_draw=(
                weight * poisson_output.probability_draw
                + (1.0 - weight) * keras_output.probability_draw
            ),
            probability_away=(
                weight * poisson_output.probability_away
                + (1.0 - weight) * keras_output.probability_away
            ),
        )

    def save(self, path: Path) -> None:
        """Guarda cada componente sin serializar TensorFlow con joblib."""

        path.mkdir(parents=True, exist_ok=True)
        self.poisson.save(path / "poisson.joblib")
        if self.keras.model is not None:
            self.keras.save(path / "keras")
        metadata = {
            "max_score": self.max_score,
            "seed": self.seed,
            "poisson_weight": self.poisson_weight,
            "validation_log_loss": self.validation_log_loss,
            "has_keras": self.keras.model is not None,
        }
        (path / "ensemble.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
