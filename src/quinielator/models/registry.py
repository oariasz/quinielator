"""Fábrica explícita de modelos disponibles."""

from __future__ import annotations

from collections.abc import Callable

from quinielator.config import ApplicationConfig
from quinielator.models.base import PredictionModel
from quinielator.models.baseline import BaselineModel
from quinielator.models.ensemble import EnsembleModel
from quinielator.models.keras_model import KerasMatchModel
from quinielator.models.sklearn_models import SklearnOutcomeModel, SklearnPoissonModel


class ModelRegistry:
    """Crea instancias nuevas para cada ventana temporal."""

    def __init__(self, config: ApplicationConfig) -> None:
        feature = config.features
        training = config.training
        seed = config.seed
        self._factories: dict[str, Callable[[], PredictionModel]] = {
            "baseline": lambda: BaselineModel(feature.max_score, seed),
            "poisson": lambda: SklearnPoissonModel(feature.max_score, seed),
            "outcome": lambda: SklearnOutcomeModel(feature.max_score, seed),
            "keras": lambda: KerasMatchModel(
                feature.max_score,
                seed,
                training.keras_epochs,
                training.keras_batch_size,
                training.keras_patience,
            ),
            "ensemble": lambda: EnsembleModel(
                feature.max_score,
                seed,
                training.keras_epochs,
                training.keras_batch_size,
                training.keras_patience,
            ),
        }

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._factories)

    def create(self, name: str) -> PredictionModel:
        try:
            return self._factories[name.lower()]()
        except KeyError as error:
            choices = ", ".join(self.names)
            raise ValueError(f"Modelo desconocido '{name}'. Opciones: {choices}") from error
