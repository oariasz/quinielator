"""Comprobaciones explícitas de calidad antes de modelar."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    """Resultado serializable de las validaciones."""

    rows: int
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["valid"] = self.valid
        return payload


class DataValidator:
    """Valida invariantes que podrían falsear el experimento."""

    REQUIRED_MATCH_COLUMNS = {
        "match_id",
        "tournament_year",
        "match_date",
        "home_team_code",
        "away_team_code",
        "home_goals_90",
        "away_goals_90",
        "home_goals_total",
        "away_goals_total",
    }

    def validate_matches(self, matches: pd.DataFrame) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        missing = self.REQUIRED_MATCH_COLUMNS.difference(matches.columns)
        if missing:
            errors.append(f"Faltan columnas: {', '.join(sorted(missing))}")
            return ValidationReport(rows=len(matches), errors=tuple(errors))
        if matches["match_id"].duplicated().any():
            errors.append("Existen match_id duplicados")
        if matches["match_date"].isna().any():
            errors.append("Existen fechas inválidas")
        if matches["home_team_code"].eq(matches["away_team_code"]).any():
            errors.append("Un equipo aparece contra sí mismo")
        score_columns = [
            "home_goals_90",
            "away_goals_90",
            "home_goals_total",
            "away_goals_total",
            "home_penalties",
            "away_penalties",
        ]
        if (matches[score_columns] < 0).any(axis=None):
            errors.append("Existen marcadores negativos")
        if (matches["home_goals_90"] > matches["home_goals_total"]).any() or (
            matches["away_goals_90"] > matches["away_goals_total"]
        ).any():
            errors.append("Un marcador a 90 minutos supera al marcador total")
        penalty_rows = matches["penalty_shootout"].eq(1)
        if (
            matches.loc[penalty_rows, "home_goals_total"]
            != matches.loc[penalty_rows, "away_goals_total"]
        ).any():
            warnings.append("Hay tandas asociadas a un marcador total no empatado")
        if not matches["match_date"].is_monotonic_increasing:
            warnings.append("Los partidos no estaban ordenados cronológicamente")
        if not ((matches["tournament_year"] >= 1930) & (matches["tournament_year"] <= 2100)).all():
            errors.append("Hay años de Mundial fuera del rango esperado")
        return ValidationReport(len(matches), tuple(errors), tuple(warnings))

    def validate_rankings(self, rankings: pd.DataFrame) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        required = {"published_on", "team_code", "rank", "points"}
        missing = required.difference(rankings.columns)
        if missing:
            errors.append(f"Faltan columnas de ranking: {', '.join(sorted(missing))}")
        elif rankings.duplicated(["published_on", "team_code"]).any():
            errors.append("Hay rankings duplicados para una misma selección y fecha")
        if not errors and (rankings["rank"] <= 0).any():
            errors.append("Hay posiciones FIFA no positivas")
        if not errors and rankings["published_on"].min().year > 1993:
            warnings.append("La fuente no cubre el inicio histórico esperado del ranking FIFA")
        return ValidationReport(len(rankings), tuple(errors), tuple(warnings))
