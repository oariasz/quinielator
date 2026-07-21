"""Repositorio local de tablas procesadas y artefactos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class DataRepository:
    """Centraliza nombres y formatos para evitar rutas dispersas."""

    RAW_FILES = {
        "matches": "world_cup_matches.csv",
        "goals": "world_cup_goals.csv",
        "teams": "world_cup_teams.csv",
        "tournaments": "world_cup_tournaments.csv",
        "world_cup_2026": "world_cup_2026.json",
        "rankings": "fifa_rankings.csv",
    }

    def __init__(self, data_directory: Path) -> None:
        self.data_directory = data_directory
        self.raw_directory = data_directory / "raw"
        self.processed_directory = data_directory / "processed"
        self.raw_directory.mkdir(parents=True, exist_ok=True)
        self.processed_directory.mkdir(parents=True, exist_ok=True)

    def raw_path(self, name: str) -> Path:
        try:
            return self.raw_directory / self.RAW_FILES[name]
        except KeyError as error:
            raise ValueError(f"Tabla desconocida: {name}") from error

    def processed_path(self, name: str) -> Path:
        return self.processed_directory / f"{name}.csv"

    def read_raw(self, name: str) -> pd.DataFrame:
        path = self.raw_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"Falta {path}. Ejecuta primero: python -m quinielator data download"
            )
        return pd.read_csv(path)

    def read_raw_json(self, name: str) -> dict[str, Any]:
        """Lee una fuente JSON conservando el mismo contrato de errores."""

        path = self.raw_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"Falta {path}. Ejecuta primero: python -m quinielator data download"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Se esperaba un objeto JSON en {path}")
        return payload

    def save_processed(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.processed_path(name)
        frame.to_csv(path, index=False)
        return path

    def read_processed(self, name: str) -> pd.DataFrame:
        path = self.processed_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"Falta {path}. Ejecuta primero: python -m quinielator data prepare"
            )
        return pd.read_csv(path, parse_dates=["match_date"] if name != "rankings" else None)

    def save_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.processed_directory / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path
