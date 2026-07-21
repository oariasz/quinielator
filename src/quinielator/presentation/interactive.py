"""Experiencia didáctica: predecir, pausar y revelar."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from quinielator.domain import MatchSign, Stage
from quinielator.services import QuinielatorApplication


class InteractivePredictionPrinter:
    """Presenta predicciones históricas sin contaminar el cálculo."""

    def __init__(
        self,
        application: QuinielatorApplication,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self.application = application
        self.input_fn = input_fn
        self.output_fn = output_fn

    def _pause(self, prompt: str) -> bool:
        return self.input_fn(prompt).strip().lower() != "q"

    def _prediction(self, row: pd.Series) -> None:
        self.output_fn("")
        self.output_fn(f"Copa Mundial {int(row['tournament_year'])} — {row['stage_name']}")
        self.output_fn(f"{row['home_team_name']} vs. {row['away_team_name']}")
        self.output_fn("")
        self.output_fn(
            f"Predicción: {row['home_team_name']} {int(row['predicted_home_goals'])}-"
            f"{int(row['predicted_away_goals'])} {row['away_team_name']}"
        )
        self.output_fn(
            f"Goles esperados: {row['expected_home_goals']:.2f} — {row['expected_away_goals']:.2f}"
        )
        self.output_fn(
            "Probabilidades a 90 minutos: "
            f"{row['home_team_name']} {row['probability_home']:.1%} | "
            f"Empate {row['probability_draw']:.1%} | "
            f"{row['away_team_name']} {row['probability_away']:.1%}"
        )
        predicted_sign = row.get(
            "predicted_sign", MatchSign.from_outcome(str(row["predicted_outcome"])).value
        )
        self.output_fn(f"Signo predicho a 90 minutos: {predicted_sign} (1/X/2)")
        advancing = (
            row["home_team_name"]
            if row["predicted_advancing_team"] == row["home_team_code"]
            else row["away_team_name"]
        )
        self.output_fn(f"Mayor probabilidad de avanzar: {advancing}")
        trained_matches = int(row["trained_matches"])
        self.output_fn(
            f"Modelo: {row['effective_model']} | Entrenado con {trained_matches} partidos"
        )

    def _actual(self, row: pd.Series) -> None:
        self.output_fn("")
        self.output_fn(
            f"Resultado a 90 minutos: {row['home_team_name']} "
            f"{int(row['actual_home_goals'])}-{int(row['actual_away_goals'])} "
            f"{row['away_team_name']}"
        )
        actual_sign = row.get(
            "actual_sign", MatchSign.from_outcome(str(row["actual_outcome"])).value
        )
        self.output_fn(f"Signo real a 90 minutos: {actual_sign} (1/X/2)")
        if int(row["actual_home_goals_total"]) != int(row["actual_home_goals"]) or int(
            row["actual_away_goals_total"]
        ) != int(row["actual_away_goals"]):
            self.output_fn(
                "Resultado tras prórroga: "
                f"{int(row['actual_home_goals_total'])}-"
                f"{int(row['actual_away_goals_total'])}"
            )
        if int(row["penalty_shootout"]):
            self.output_fn(
                f"Penaltis: {int(row['actual_home_penalties'])}-{int(row['actual_away_penalties'])}"
            )
        exact = int(row["predicted_home_goals"]) == int(row["actual_home_goals"]) and int(
            row["predicted_away_goals"]
        ) == int(row["actual_away_goals"])
        outcome_hit = row["predicted_outcome"] == row["actual_outcome"]
        self.output_fn(f"Resultado 1X2 acertado: {'sí' if outcome_hit else 'no'}")
        actual_advancing = str(row["actual_advancing_team"])
        if actual_advancing:
            self.output_fn(
                "Selección que avanzó acertada: "
                f"{'sí' if row['predicted_advancing_team'] == actual_advancing else 'no'}"
            )
        self.output_fn(f"Marcador exacto: {'sí' if exact else 'no'}")

    def run(
        self,
        *,
        stage: str = "final",
        start_year: int | None = None,
        end_year: int | None = None,
        model_name: str = "ensemble",
        evaluation_mode: str = "strict_editions",
    ) -> None:
        normalized_stage = Stage.parse(stage)
        if normalized_stage is Stage.OTHER:
            raise ValueError("Fase desconocida. Usa final, semifinal/seminfinal/semis o cuartos.")
        predictions, metrics = self.application.load_or_evaluate_predictions(
            model_name=model_name,
            start_year=start_year,
            end_year=end_year,
            mode=evaluation_mode,
        )
        predictions = predictions.copy()
        selected_years = sorted(int(year) for year in predictions["tournament_year"].unique())
        shown = 0
        for year in selected_years:
            tournament = predictions[predictions["tournament_year"].eq(year)]
            stage_matches = tournament[tournament["stage"].eq(normalized_stage.value)]
            if stage_matches.empty:
                if normalized_stage is Stage.FINAL and year == 1950:
                    self.output_fn(
                        "Mundial 1950: no tuvo una final oficial; se omite sin inventarla."
                    )
                continue
            for _, row in stage_matches.iterrows():
                self._prediction(row)
                if not self._pause(
                    "\nPulsa Enter para revelar el resultado o escribe q para salir: "
                ):
                    return
                self._actual(row)
                shown += 1
                if not self._pause("\nPulsa Enter para continuar o escribe q para salir: "):
                    return
            metric_row = metrics[metrics["tournament_year"].eq(year)]
            if not metric_row.empty:
                item = metric_row.iloc[0]
                self.output_fn("")
                self.output_fn(f"Evaluación completa del Mundial {year}")
                self.output_fn(f"Partidos predichos: {int(item['matches'])}")
                self.output_fn(f"Accuracy 1X2: {item['outcome_accuracy']:.1%}")
                self.output_fn(f"Marcadores exactos: {item['exact_score_accuracy']:.1%}")
                self.output_fn(f"MAE de goles: {item['goals_mae']:.3f}")
                self.output_fn(f"Log loss: {item['log_loss']:.3f}")
        if shown == 0:
            self.output_fn("No se encontraron partidos para la fase y rango solicitados.")


def print_predictions(
    stage: str = "final",
    start_year: int | None = None,
    end_year: int | None = None,
    model_name: str = "ensemble",
    evaluation_mode: str = "strict_editions",
) -> None:
    """Fachada pública de la experiencia interactiva."""

    InteractivePredictionPrinter(QuinielatorApplication()).run(
        stage=stage,
        start_year=start_year,
        end_year=end_year,
        model_name=model_name,
        evaluation_mode=evaluation_mode,
    )
