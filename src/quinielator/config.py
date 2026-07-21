"""Configuración tipada y reproducible del proyecto."""

from __future__ import annotations

import os
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProjectPaths:
    """Rutas de trabajo resueltas desde la raíz del repositorio."""

    root: Path
    data: Path
    models: Path
    reports: Path

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        return self.data / "processed"


@dataclass(frozen=True)
class SourceConfig:
    """URLs de las fuentes descargables."""

    world_cup_matches_url: str
    world_cup_goals_url: str
    world_cup_teams_url: str
    world_cup_tournaments_url: str
    world_cup_2026_url: str
    fifa_rankings_url: str


@dataclass(frozen=True)
class FeatureConfig:
    """Parámetros de ingeniería de variables."""

    rolling_window: int = 5
    elo_initial: float = 1500.0
    elo_k_factor: float = 30.0
    elo_home_advantage: float = 0.0
    max_score: int = 8


@dataclass(frozen=True)
class TrainingConfig:
    """Parámetros de entrenamiento y evaluación."""

    minimum_training_matches: int = 200
    evaluation_mode: str = "strict_editions"
    start_year: int = 1958
    keras_epochs: int = 80
    keras_batch_size: int = 32
    keras_patience: int = 10


@dataclass(frozen=True)
class ApplicationConfig:
    """Configuración raíz de Quinielator."""

    seed: int
    paths: ProjectPaths
    sources: SourceConfig
    features: FeatureConfig
    training: TrainingConfig

    @classmethod
    def load(cls, path: Path | str = "config/default.toml") -> ApplicationConfig:
        """Carga la configuración TOML y resuelve rutas contra el repositorio."""

        config_path = Path(path).resolve()
        with config_path.open("rb") as handle:
            raw: dict[str, Any] = tomllib.load(handle)

        root = config_path.parent.parent
        paths_raw = raw["paths"]
        source_raw = raw["sources"]
        feature_raw = raw["features"]
        training_raw = raw["training"]
        return cls(
            seed=int(raw["project"]["seed"]),
            paths=ProjectPaths(
                root=root,
                data=root / str(paths_raw["data_dir"]),
                models=root / str(paths_raw["models_dir"]),
                reports=root / str(paths_raw["reports_dir"]),
            ),
            sources=SourceConfig(**source_raw),
            features=FeatureConfig(**feature_raw),
            training=TrainingConfig(**training_raw),
        )

    def ensure_directories(self) -> None:
        """Crea únicamente los directorios locales necesarios."""

        for directory in (
            self.paths.raw,
            self.paths.data / "interim",
            self.paths.processed,
            self.paths.models,
            self.paths.reports,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def apply_seeds(self) -> None:
        """Fija las semillas disponibles; TensorFlow se trata como dependencia opcional."""

        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
        cache_directory = self.paths.reports / ".cache"
        cache_directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_directory))
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        random.seed(self.seed)
        np.random.seed(self.seed)
        # TensorFlow se inicializa de forma perezosa dentro de KerasMatchModel;
        # importarlo aquí haría que hasta `data validate` cargase un runtime pesado.
