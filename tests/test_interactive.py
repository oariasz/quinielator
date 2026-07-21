from __future__ import annotations

from typing import Any

import pandas as pd

from quinielator.presentation import InteractivePredictionPrinter


class _FakeApplication:
    def load_or_evaluate_predictions(self, **_: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        predictions = pd.DataFrame(
            [
                {
                    "tournament_year": 2022,
                    "stage": "final",
                    "stage_name": "final",
                    "home_team_name": "Argentina",
                    "away_team_name": "France",
                    "home_team_code": "ARG",
                    "away_team_code": "FRA",
                    "predicted_home_goals": 1,
                    "predicted_away_goals": 1,
                    "expected_home_goals": 1.4,
                    "expected_away_goals": 1.3,
                    "probability_home": 0.4,
                    "probability_draw": 0.3,
                    "probability_away": 0.3,
                    "predicted_advancing_team": "ARG",
                    "effective_model": "poisson",
                    "trained_matches": 900,
                    "actual_home_goals": 2,
                    "actual_away_goals": 2,
                    "actual_home_goals_total": 3,
                    "actual_away_goals_total": 3,
                    "actual_home_penalties": 4,
                    "actual_away_penalties": 2,
                    "penalty_shootout": 1,
                    "predicted_outcome": "D",
                    "actual_outcome": "D",
                    "actual_advancing_team": "ARG",
                }
            ]
        )
        metrics = pd.DataFrame(
            [
                {
                    "tournament_year": 2022,
                    "matches": 64,
                    "outcome_accuracy": 0.55,
                    "exact_score_accuracy": 0.12,
                    "goals_mae": 0.9,
                    "log_loss": 1.02,
                }
            ]
        )
        return predictions, metrics


def test_interactive_reveals_only_after_enter() -> None:
    answers = iter(["", "q"])
    output: list[str] = []
    printer = InteractivePredictionPrinter(
        _FakeApplication(),  # type: ignore[arg-type]
        input_fn=lambda _: next(answers),
        output_fn=output.append,
    )
    printer.run(stage="final")
    combined = "\n".join(output)
    assert "Predicción: Argentina 1-1 France" in combined
    assert "Resultado a 90 minutos: Argentina 2-2 France" in combined
    assert "Penaltis: 4-2" in combined
