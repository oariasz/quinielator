"""Modelos intercambiables con una salida probabilística común."""

from quinielator.models.base import BatchPrediction, PredictionModel
from quinielator.models.baseline import BaselineModel
from quinielator.models.ensemble import EnsembleModel
from quinielator.models.keras_model import KerasMatchModel
from quinielator.models.registry import ModelRegistry
from quinielator.models.sklearn_models import SklearnOutcomeModel, SklearnPoissonModel

__all__ = [
    "BaselineModel",
    "BatchPrediction",
    "EnsembleModel",
    "KerasMatchModel",
    "ModelRegistry",
    "PredictionModel",
    "SklearnOutcomeModel",
    "SklearnPoissonModel",
]
