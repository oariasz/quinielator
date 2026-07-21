"""Contratos y utilidades probabilísticas compartidas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson


@dataclass(frozen=True)
class BatchPrediction:
    """Predicciones vectorizadas en el mismo orden que la entrada."""

    expected_home_goals: np.ndarray
    expected_away_goals: np.ndarray
    predicted_home_goals: np.ndarray
    predicted_away_goals: np.ndarray
    probability_home: np.ndarray
    probability_draw: np.ndarray
    probability_away: np.ndarray


def score_probabilities(
    expected_home: np.ndarray,
    expected_away: np.ndarray,
    max_score: int,
) -> BatchPrediction:
    """Convierte intensidades Poisson en resultados y marcadores coherentes."""

    expected_home = np.clip(np.asarray(expected_home, dtype=float), 0.05, 5.0)
    expected_away = np.clip(np.asarray(expected_away, dtype=float), 0.05, 5.0)
    count = len(expected_home)
    predicted_home = np.zeros(count, dtype=int)
    predicted_away = np.zeros(count, dtype=int)
    probability_home = np.zeros(count, dtype=float)
    probability_draw = np.zeros(count, dtype=float)
    probability_away = np.zeros(count, dtype=float)
    values = np.arange(max_score + 1)
    for index, (home_lambda, away_lambda) in enumerate(
        zip(expected_home, expected_away, strict=True)
    ):
        home_lambda = float(home_lambda)
        away_lambda = float(away_lambda)
        home_pmf = poisson.pmf(values, home_lambda)
        away_pmf = poisson.pmf(values, away_lambda)
        home_pmf[-1] += max(0.0, 1.0 - float(home_pmf.sum()))
        away_pmf[-1] += max(0.0, 1.0 - float(away_pmf.sum()))
        matrix = np.outer(home_pmf, away_pmf)
        matrix /= matrix.sum()
        predicted_home[index], predicted_away[index] = np.unravel_index(
            int(np.argmax(matrix)), matrix.shape
        )
        probability_home[index] = float(np.tril(matrix, k=-1).sum())
        probability_draw[index] = float(np.trace(matrix))
        probability_away[index] = float(np.triu(matrix, k=1).sum())
    return BatchPrediction(
        expected_home_goals=np.asarray(expected_home, dtype=float),
        expected_away_goals=np.asarray(expected_away, dtype=float),
        predicted_home_goals=predicted_home,
        predicted_away_goals=predicted_away,
        probability_home=probability_home,
        probability_draw=probability_draw,
        probability_away=probability_away,
    )


class PredictionModel(ABC):
    """Interfaz común de todos los modelos."""

    name: str

    def __init__(self, max_score: int = 8, seed: int = 42) -> None:
        self.max_score = max_score
        self.seed = seed

    @abstractmethod
    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> PredictionModel:
        """Ajusta el modelo con partidos estrictamente anteriores."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> BatchPrediction:
        """Predice probabilidades y marcador para cada fila."""

    def save(self, path: Path) -> None:
        """Serializa modelos compatibles con joblib."""

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> PredictionModel:
        """Carga un modelo serializado."""

        model = joblib.load(path)
        if not isinstance(model, PredictionModel):
            raise TypeError(f"{path} no contiene un PredictionModel")
        return model
