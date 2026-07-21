"""Entrena modelos para uso local después del backtesting."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from quinielator.config import ApplicationConfig
from quinielator.features import TemporalDatasetBuilder
from quinielator.models import EnsembleModel, KerasMatchModel, ModelRegistry


class ModelTrainer:
    """Ajusta un modelo con toda la historia y guarda trazabilidad."""

    def __init__(self, config: ApplicationConfig, registry: ModelRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or ModelRegistry(config)

    def train(self, dataset: pd.DataFrame, model_name: str) -> Path:
        feature_columns = TemporalDatasetBuilder.feature_columns(dataset)
        targets = pd.DataFrame(
            {
                "home_goals": dataset["home_goals_90"],
                "away_goals": dataset["away_goals_90"],
                "outcome": dataset["target_outcome"],
            }
        )
        model = self.registry.create(model_name)
        model.fit(dataset[feature_columns], targets)
        destination = self.config.paths.models / model_name
        if isinstance(model, (KerasMatchModel, EnsembleModel)):
            model.save(destination)
        else:
            destination = destination.with_suffix(".joblib")
            model.save(destination)
        metadata = {
            "model_name": model_name,
            "trained_at": datetime.now(UTC).isoformat(),
            "matches": len(dataset),
            "first_match": str(dataset["match_date"].min()),
            "last_match": str(dataset["match_date"].max()),
            "feature_columns": feature_columns,
            "seed": self.config.seed,
        }
        metadata_path = self.config.paths.models / f"{model_name}.metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return destination
