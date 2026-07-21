from __future__ import annotations

import pandas as pd
import pytest

from quinielator.evaluation import CalibrationEvaluator, MetricsCalculator


def test_metrics_distinguish_outcome_and_exact_score() -> None:
    predictions = pd.DataFrame(
        [
            {
                "actual_outcome": "H",
                "predicted_outcome": "H",
                "probability_away": 0.1,
                "probability_draw": 0.2,
                "probability_home": 0.7,
                "actual_advancing_team": "AAA",
                "predicted_advancing_team": "AAA",
                "actual_home_goals": 2,
                "actual_away_goals": 1,
                "expected_home_goals": 1.5,
                "expected_away_goals": 0.9,
                "predicted_home_goals": 1,
                "predicted_away_goals": 0,
            }
        ]
    )
    metrics = MetricsCalculator().calculate(predictions)
    assert metrics.outcome_accuracy == 1.0
    assert metrics.exact_score_accuracy == 0.0
    assert metrics.advancing_accuracy == 1.0
    assert metrics.extra["calibration_error"] == pytest.approx(0.3)
    assert metrics.extra["accuracy_ci_lower"] < metrics.outcome_accuracy


def test_calibration_table_accounts_for_all_predictions() -> None:
    predictions = pd.DataFrame(
        {
            "probability_away": [0.1, 0.6],
            "probability_draw": [0.2, 0.2],
            "probability_home": [0.7, 0.2],
            "predicted_outcome": ["H", "A"],
            "actual_outcome": ["H", "D"],
        }
    )
    table = CalibrationEvaluator().table(predictions)
    assert table["matches"].sum() == 2
