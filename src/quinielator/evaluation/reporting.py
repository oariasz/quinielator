"""Reportes legibles y gráficas sencillas del experimento."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quinielator.evaluation.metrics import CalibrationEvaluator, MetricsCalculator


class ReportGenerator:
    """Genera artefactos reproducibles a partir de métricas ya calculadas."""

    def __init__(self, reports_directory: Path) -> None:
        self.reports_directory = reports_directory
        self.reports_directory.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        predictions: pd.DataFrame,
        metrics: pd.DataFrame,
        model_name: str,
    ) -> tuple[Path, Path]:
        prediction_path = self.reports_directory / f"predictions_{model_name}.csv"
        metrics_path = self.reports_directory / f"metrics_{model_name}.csv"
        predictions.to_csv(prediction_path, index=False)
        metrics.to_csv(metrics_path, index=False)
        predictions.to_json(
            self.reports_directory / f"predictions_{model_name}.json",
            orient="records",
            indent=2,
            force_ascii=False,
        )

        labels = ["A", "D", "H"]
        confusion = pd.crosstab(
            predictions["actual_outcome"],
            predictions["predicted_outcome"],
            rownames=["real"],
            colnames=["predicho"],
        ).reindex(index=labels, columns=labels, fill_value=0)
        confusion.to_csv(self.reports_directory / f"confusion_{model_name}.csv")
        CalibrationEvaluator().table(predictions).to_csv(
            self.reports_directory / f"calibration_{model_name}.csv", index=False
        )
        phase_records: list[dict[str, float | int | str]] = []
        calculator = MetricsCalculator()
        for stage, group in predictions.groupby("stage", sort=True):
            phase_metrics = calculator.calculate(group)
            phase_records.append(
                {
                    "stage": str(stage),
                    "matches": phase_metrics.matches,
                    "outcome_accuracy": phase_metrics.outcome_accuracy,
                    "exact_score_accuracy": phase_metrics.exact_score_accuracy,
                    "goals_mae": phase_metrics.goals_mae,
                    "log_loss": phase_metrics.log_loss,
                    "brier_score": phase_metrics.brier_score,
                    "calibration_error": phase_metrics.extra["calibration_error"],
                }
            )
        pd.DataFrame.from_records(phase_records).to_csv(
            self.reports_directory / f"metrics_by_phase_{model_name}.csv", index=False
        )

        self._write_accuracy_svg(metrics, model_name)

        markdown_path = self.reports_directory / f"report_{model_name}.md"
        best = metrics.loc[metrics["outcome_accuracy"].idxmax()]
        latest = metrics.iloc[-1]
        best_year = int(float(str(best["tournament_year"])))
        best_accuracy = float(str(best["outcome_accuracy"]))
        latest_year = int(float(str(latest["tournament_year"])))
        cumulative_accuracy = float(str(latest["cumulative_outcome_accuracy"]))
        latest_log_loss = float(str(latest["log_loss"]))
        latest_coverage = float(str(latest["fifa_ranking_coverage"]))
        latest_ci_lower = float(str(latest["accuracy_ci_lower"]))
        latest_ci_upper = float(str(latest["accuracy_ci_upper"]))
        latest_calibration = float(str(latest["calibration_error"]))
        markdown = f"""# Evaluación de {model_name}

Este informe usa backtesting temporal. Cada edición se predice con información anterior.
La curva no tiene por qué ser ascendente: más historia no garantiza mejora monótona.

## Resumen

- Mundiales evaluados: {len(metrics)}
- Partidos predichos: {len(predictions)}
- Mejor edición por accuracy 1X2: {best_year} ({best_accuracy:.1%})
- Última edición: {latest_year}
- Accuracy acumulada final: {cumulative_accuracy:.1%}
- Log loss de la última edición: {latest_log_loss:.3f}
- Intervalo Wilson 95% de accuracy (última edición): {latest_ci_lower:.1%}–{latest_ci_upper:.1%}
- Error de calibración de la última edición: {latest_calibration:.3f}
- Cobertura FIFA de la última edición: {latest_coverage:.1%}

Consulta `metrics_{model_name}.csv` para el detalle y `accuracy_{model_name}.svg` para la gráfica.
"""
        markdown_path.write_text(markdown, encoding="utf-8")
        return prediction_path, metrics_path

    def _write_accuracy_svg(self, metrics: pd.DataFrame, model_name: str) -> Path:
        """Dibuja una curva SVG sin cargar un backend gráfico junto a TensorFlow."""

        width, height = 960, 480
        left, right, top, bottom = 70, 30, 55, 60
        plot_width = width - left - right
        plot_height = height - top - bottom
        years = metrics["tournament_year"].astype(int).tolist()
        accuracy = metrics["outcome_accuracy"].astype(float).tolist()
        cumulative = metrics["cumulative_outcome_accuracy"].astype(float).tolist()

        def point(index: int, value: float) -> tuple[float, float]:
            denominator = max(len(years) - 1, 1)
            x = left + plot_width * index / denominator
            y = top + plot_height * (1.0 - value)
            return x, y

        accuracy_points = " ".join(
            f"{x:.1f},{y:.1f}" for x, y in (point(i, value) for i, value in enumerate(accuracy))
        )
        cumulative_points = " ".join(
            f"{x:.1f},{y:.1f}" for x, y in (point(i, value) for i, value in enumerate(cumulative))
        )
        grid = []
        for step in range(0, 11, 2):
            value = step / 10
            y = top + plot_height * (1.0 - value)
            grid.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
                'stroke="#d9e2ec" stroke-width="1" />'
            )
            grid.append(
                f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-size="12">{value:.0%}</text>'
            )
        labels = []
        for index, year in enumerate(years):
            x, _ = point(index, 0.0)
            labels.append(
                f'<text x="{x:.1f}" y="{height - 28}" text-anchor="middle" '
                f'font-size="11">{year}</text>'
            )
        header = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        )
        title = (
            f'<text x="{width / 2:.0f}" y="30" text-anchor="middle" font-size="20" '
            f'font-weight="600">Evolución temporal — {model_name}</text>'
        )
        svg = f"""{header}
<rect width="100%" height="100%" fill="white" />
{title}
{"".join(grid)}
<polyline points="{accuracy_points}" fill="none" stroke="#d97706" stroke-width="3" />
<polyline points="{cumulative_points}" fill="none" stroke="#2563eb" stroke-width="3" />
{"".join(labels)}
<text x="{left}" y="{height - 8}" font-size="12" fill="#d97706">● Accuracy por Mundial</text>
<text x="{left + 190}" y="{height - 8}" font-size="12" fill="#2563eb">● Media acumulada</text>
</svg>
"""
        path = self.reports_directory / f"accuracy_{model_name}.svg"
        path.write_text(svg, encoding="utf-8")
        return path

    def generate_comparison(self, metrics_by_model: dict[str, pd.DataFrame]) -> Path:
        """Resume todos los modelos con ponderación por número de partidos."""

        records: list[dict[str, float | int | str]] = []
        for model_name, metrics in metrics_by_model.items():
            matches = int(metrics["matches"].sum())
            weighted = metrics["matches"].astype(float)
            records.append(
                {
                    "model": model_name,
                    "matches": matches,
                    "outcome_accuracy": float(metrics["cumulative_outcome_accuracy"].iloc[-1]),
                    "exact_score_accuracy": float(
                        (metrics["exact_score_accuracy"] * weighted).sum() / matches
                    ),
                    "goals_mae": float((metrics["goals_mae"] * weighted).sum() / matches),
                    "log_loss": float((metrics["log_loss"] * weighted).sum() / matches),
                    "brier_score": float((metrics["brier_score"] * weighted).sum() / matches),
                    "calibration_error": float(
                        (metrics["calibration_error"] * weighted).sum() / matches
                    ),
                }
            )
        comparison = pd.DataFrame.from_records(records).sort_values(
            ["log_loss", "outcome_accuracy"], ascending=[True, False]
        )
        path = self.reports_directory / "model_comparison.csv"
        comparison.to_csv(path, index=False)
        lines = [
            "# Comparación temporal de modelos",
            "",
            "Las métricas agregan los mismos partidos. Menor log loss y Brier es mejor.",
            "",
            "| Modelo | Accuracy 1X2 | Marcador exacto | MAE goles | Log loss | Brier |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for _, row in comparison.iterrows():
            lines.append(
                f"| {row['model']} | {row['outcome_accuracy']:.1%} | "
                f"{row['exact_score_accuracy']:.1%} | {row['goals_mae']:.3f} | "
                f"{row['log_loss']:.3f} | {row['brier_score']:.3f} |"
            )
        (self.reports_directory / "model_comparison.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        return path
