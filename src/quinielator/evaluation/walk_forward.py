"""Backtesting expanding-window por edición o por partido."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd

from quinielator.config import ApplicationConfig
from quinielator.evaluation.metrics import MetricsCalculator
from quinielator.features import TemporalDatasetBuilder
from quinielator.models import BatchPrediction, ModelRegistry, PredictionModel


class WorldCupEvaluator:
    """Predice cronológicamente y nunca entrena con el Mundial objetivo futuro."""

    MODES = {"strict_editions", "walk_forward"}

    def __init__(self, config: ApplicationConfig, registry: ModelRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or ModelRegistry(config)
        self.metrics = MetricsCalculator()

    @staticmethod
    def _targets(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "home_goals": frame["home_goals_90"].astype(float),
                "away_goals": frame["away_goals_90"].astype(float),
                "outcome": frame["target_outcome"].astype(str),
            },
            index=frame.index,
        )

    def _model_for(self, name: str, training_rows: int) -> tuple[PredictionModel, str]:
        if training_rows < self.config.training.minimum_training_matches and name != "baseline":
            return self.registry.create("baseline"), "baseline_fallback"
        return self.registry.create(name), name

    def _records(
        self,
        frame: pd.DataFrame,
        output: BatchPrediction,
        requested_model: str,
        effective_model: str,
        trained_matches: int,
        poisson_weight: float,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for position, (_, row) in enumerate(frame.iterrows()):
            home_probability = float(output.probability_home[position])
            away_probability = float(output.probability_away[position])
            predicted_outcome = max(
                (
                    (home_probability, "H"),
                    (float(output.probability_draw[position]), "D"),
                    (away_probability, "A"),
                )
            )[1]
            predicted_advancing = (
                str(row["home_team_code"])
                if home_probability >= away_probability
                else str(row["away_team_code"])
            )
            match_date = pd.Timestamp(row["match_date"])
            records.append(
                {
                    "match_id": str(row["match_id"]),
                    "tournament_year": int(row["tournament_year"]),
                    "match_date": match_date.date().isoformat(),
                    "as_of_date": (match_date - timedelta(microseconds=1)).isoformat(),
                    "stage": str(row["stage"]),
                    "stage_name": str(row["stage_name"]),
                    "home_team_code": str(row["home_team_code"]),
                    "home_team_name": str(row["home_team_name"]),
                    "away_team_code": str(row["away_team_code"]),
                    "away_team_name": str(row["away_team_name"]),
                    "expected_home_goals": float(output.expected_home_goals[position]),
                    "expected_away_goals": float(output.expected_away_goals[position]),
                    "predicted_home_goals": int(output.predicted_home_goals[position]),
                    "predicted_away_goals": int(output.predicted_away_goals[position]),
                    "probability_home": home_probability,
                    "probability_draw": float(output.probability_draw[position]),
                    "probability_away": away_probability,
                    "predicted_outcome": predicted_outcome,
                    "predicted_advancing_team": predicted_advancing,
                    "actual_home_goals": int(row["home_goals_90"]),
                    "actual_away_goals": int(row["away_goals_90"]),
                    "actual_home_goals_total": int(row["home_goals_total"]),
                    "actual_away_goals_total": int(row["away_goals_total"]),
                    "actual_home_penalties": int(row["home_penalties"]),
                    "actual_away_penalties": int(row["away_penalties"]),
                    "actual_outcome": str(row["target_outcome"]),
                    "actual_advancing_team": str(row["actual_advancing_team"]),
                    "penalty_shootout": int(row["penalty_shootout"]),
                    "fifa_ranking_available": float(row["fifa_ranking_available"]),
                    "model_name": requested_model,
                    "effective_model": effective_model,
                    "trained_matches": trained_matches,
                    "poisson_weight": poisson_weight,
                }
            )
        return records

    def evaluate(
        self,
        dataset: pd.DataFrame,
        *,
        model_name: str = "poisson",
        start_year: int | None = None,
        end_year: int | None = None,
        mode: str = "strict_editions",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Ejecuta el backtest y devuelve predicciones y métricas por edición."""

        if mode not in self.MODES:
            raise ValueError(f"Modo desconocido: {mode}. Opciones: {', '.join(sorted(self.MODES))}")
        feature_columns = TemporalDatasetBuilder.feature_columns(dataset)
        first_year = start_year or self.config.training.start_year
        last_year = end_year or int(dataset["tournament_year"].max())
        years = [
            int(year)
            for year in sorted(dataset["tournament_year"].unique())
            if first_year <= int(year) <= last_year
        ]
        records: list[dict[str, Any]] = []
        for year in years:
            target = dataset[dataset["tournament_year"].eq(year)].copy()
            history = dataset[dataset["tournament_year"].lt(year)].copy()
            if history.empty or target.empty:
                continue
            if mode == "strict_editions":
                model, effective = self._model_for(model_name, len(history))
                model.fit(history[feature_columns], self._targets(history))
                output = model.predict(target[feature_columns])
                poisson_weight = float(getattr(model, "poisson_weight", float("nan")))
                records.extend(
                    self._records(
                        target,
                        output,
                        model_name,
                        effective,
                        len(history),
                        poisson_weight,
                    )
                )
                continue
            for index in target.index:
                before_current = dataset.loc[dataset.index < index]
                model, effective = self._model_for(model_name, len(before_current))
                model.fit(before_current[feature_columns], self._targets(before_current))
                current = dataset.loc[[index]]
                output = model.predict(current[feature_columns])
                poisson_weight = float(getattr(model, "poisson_weight", float("nan")))
                records.extend(
                    self._records(
                        current,
                        output,
                        model_name,
                        effective,
                        len(before_current),
                        poisson_weight,
                    )
                )
        predictions = pd.DataFrame.from_records(records)
        if predictions.empty:
            raise ValueError("El rango solicitado no produjo predicciones")
        return predictions, self.metrics.by_tournament(predictions)
