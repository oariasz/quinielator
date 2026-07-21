"""Métricas complementarias para no reducir todo a una sola precisión."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error

from quinielator.domain import EvaluationMetrics


class CalibrationEvaluator:
    """Resume si la confianza declarada coincide con la frecuencia de acierto."""

    def table(self, predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
        probabilities = predictions[
            ["probability_away", "probability_draw", "probability_home"]
        ].to_numpy(dtype=float)
        confidence = probabilities.max(axis=1)
        correct = (
            predictions["predicted_outcome"].astype(str)
            == predictions["actual_outcome"].astype(str)
        ).to_numpy(dtype=float)
        edges = np.linspace(0.0, 1.0, bins + 1)
        records: list[dict[str, float | int]] = []
        for index in range(bins):
            lower = float(edges[index])
            upper = float(edges[index + 1])
            selected = (confidence >= lower) & (
                confidence <= upper if index == bins - 1 else confidence < upper
            )
            count = int(selected.sum())
            if count == 0:
                continue
            records.append(
                {
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "matches": count,
                    "mean_confidence": float(confidence[selected].mean()),
                    "observed_accuracy": float(correct[selected].mean()),
                }
            )
        return pd.DataFrame.from_records(records)

    def expected_calibration_error(self, predictions: pd.DataFrame, bins: int = 10) -> float:
        table = self.table(predictions, bins)
        total = int(table["matches"].sum())
        if total == 0:
            return float("nan")
        gaps = (table["mean_confidence"] - table["observed_accuracy"]).abs()
        return float((gaps * table["matches"] / total).sum())


class MetricsCalculator:
    """Calcula métricas sobre el formato tabular de predicciones."""

    LABELS = ["A", "D", "H"]

    def __init__(self) -> None:
        self.calibration = CalibrationEvaluator()

    @staticmethod
    def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
        proportion = successes / total
        denominator = 1.0 + z**2 / total
        center = (proportion + z**2 / (2.0 * total)) / denominator
        margin = (
            z
            * np.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2))
            / denominator
        )
        return float(center - margin), float(center + margin)

    def calculate(self, predictions: pd.DataFrame) -> EvaluationMetrics:
        if predictions.empty:
            raise ValueError("No hay predicciones para evaluar")
        probabilities = predictions[
            ["probability_away", "probability_draw", "probability_home"]
        ].to_numpy(dtype=float)
        probabilities = np.clip(probabilities, 1e-9, 1.0)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        actual_outcome = predictions["actual_outcome"].astype(str)
        predicted_outcome = predictions["predicted_outcome"].astype(str)
        actual_one_hot = np.column_stack(
            [(actual_outcome.to_numpy() == label).astype(float) for label in self.LABELS]
        )
        brier = float(np.mean(np.sum((probabilities - actual_one_hot) ** 2, axis=1)))
        advancing = predictions[predictions["actual_advancing_team"].fillna("").ne("")]
        advancing_accuracy: float | None = None
        if not advancing.empty:
            advancing_accuracy = float(
                (advancing["predicted_advancing_team"] == advancing["actual_advancing_team"]).mean()
            )
        home_mae = float(
            mean_absolute_error(
                predictions["actual_home_goals"], predictions["expected_home_goals"]
            )
        )
        away_mae = float(
            mean_absolute_error(
                predictions["actual_away_goals"], predictions["expected_away_goals"]
            )
        )
        exact = (predictions["predicted_home_goals"] == predictions["actual_home_goals"]) & (
            predictions["predicted_away_goals"] == predictions["actual_away_goals"]
        )
        correct_outcomes = int((predicted_outcome == actual_outcome).sum())
        ci_lower, ci_upper = self._wilson_interval(correct_outcomes, len(predictions))
        return EvaluationMetrics(
            matches=len(predictions),
            outcome_accuracy=float((predicted_outcome == actual_outcome).mean()),
            exact_score_accuracy=float(exact.mean()),
            advancing_accuracy=advancing_accuracy,
            goals_mae=(home_mae + away_mae) / 2.0,
            log_loss=float(log_loss(actual_outcome, probabilities, labels=self.LABELS)),
            brier_score=brier,
            home_goal_mae=home_mae,
            away_goal_mae=away_mae,
            extra={
                "accuracy_ci_lower": ci_lower,
                "accuracy_ci_upper": ci_upper,
                "calibration_error": self.calibration.expected_calibration_error(predictions),
            },
        )

    def by_tournament(self, predictions: pd.DataFrame) -> pd.DataFrame:
        records: list[dict[str, float | int | str]] = []
        cumulative_correct = 0
        cumulative_matches = 0
        for year, group in predictions.groupby("tournament_year", sort=True):
            metrics = self.calculate(group)
            cumulative_correct += int((group["predicted_outcome"] == group["actual_outcome"]).sum())
            cumulative_matches += len(group)
            records.append(
                {
                    "tournament_year": int(str(year)),
                    "model_name": str(group["model_name"].iloc[0]),
                    "effective_model": str(group["effective_model"].iloc[0]),
                    "matches": metrics.matches,
                    "trained_matches": int(group["trained_matches"].min()),
                    "outcome_accuracy": metrics.outcome_accuracy,
                    "cumulative_outcome_accuracy": cumulative_correct / cumulative_matches,
                    "exact_score_accuracy": metrics.exact_score_accuracy,
                    "advancing_accuracy": (
                        metrics.advancing_accuracy
                        if metrics.advancing_accuracy is not None
                        else float("nan")
                    ),
                    "goals_mae": metrics.goals_mae,
                    "log_loss": metrics.log_loss,
                    "brier_score": metrics.brier_score,
                    "calibration_error": metrics.extra["calibration_error"],
                    "accuracy_ci_lower": metrics.extra["accuracy_ci_lower"],
                    "accuracy_ci_upper": metrics.extra["accuracy_ci_upper"],
                    "fifa_ranking_coverage": float(group["fifa_ranking_available"].mean()),
                    "poisson_weight": float(group["poisson_weight"].mean()),
                }
            )
        return pd.DataFrame.from_records(records)
