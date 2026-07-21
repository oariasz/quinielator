"""Importancia por permutación compatible con todos los modelos de Quinielator."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from quinielator.config import ApplicationConfig
from quinielator.features import TemporalDatasetBuilder
from quinielator.models import BatchPrediction, ModelRegistry, PredictionModel


@dataclass(frozen=True)
class FeatureImportanceResult:
    """Resultado auditable de una explicación posterior al backtest."""

    features: pd.DataFrame
    groups: pd.DataFrame
    baseline_log_loss: float
    model_name: str
    tournament_year: int
    trained_matches: int
    test_matches: int
    repeats: int
    poisson_weight: float


@dataclass(frozen=True)
class FeatureImportanceArtifacts:
    """Archivos generados por el reporte de importancia."""

    features_csv: Path
    groups_csv: Path
    markdown: Path
    svg: Path


class FeatureImportanceAnalyzer:
    """Mide cuánto aumenta el log loss al romper una señal prepartido."""

    LABELS = ["A", "D", "H"]

    def __init__(
        self,
        config: ApplicationConfig,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or ModelRegistry(config)

    def analyze(
        self,
        dataset: pd.DataFrame,
        *,
        model_name: str,
        tournament_year: int,
        repeats: int = 3,
    ) -> FeatureImportanceResult:
        """Entrena con el pasado y permuta únicamente el Mundial explicado."""

        if repeats < 1:
            raise ValueError("repeats debe ser al menos 1")
        history = dataset[dataset["tournament_year"].lt(tournament_year)].copy()
        target = dataset[dataset["tournament_year"].eq(tournament_year)].copy()
        if history.empty or target.empty:
            raise ValueError(f"No hay historia y objetivo suficientes para {tournament_year}")

        feature_columns = TemporalDatasetBuilder.feature_columns(dataset)
        train_targets = pd.DataFrame(
            {
                "home_goals": history["home_goals_90"].astype(float),
                "away_goals": history["away_goals_90"].astype(float),
                "outcome": history["target_outcome"].astype(str),
            },
            index=history.index,
        )
        model = self.registry.create(model_name)
        model.fit(history[feature_columns], train_targets)
        target_features = target[feature_columns].copy()
        actual = target["target_outcome"].astype(str)
        baseline = self._loss(actual, model.predict(target_features))
        rng = np.random.default_rng(self.config.seed)

        feature_records: list[dict[str, float | int | str]] = []
        for feature in feature_columns:
            deltas = self._permutation_deltas(
                model=model,
                features=target_features,
                actual=actual,
                columns=[feature],
                baseline=baseline,
                repeats=repeats,
                rng=rng,
            )
            feature_group, source = self.classify(feature)
            feature_records.append(
                {
                    "feature": feature,
                    "group": feature_group,
                    "source": source,
                    "importance_log_loss": float(np.mean(deltas)),
                    "importance_std": float(np.std(deltas)),
                }
            )

        features = self._rank(pd.DataFrame.from_records(feature_records), "feature")
        group_records: list[dict[str, float | int | str]] = []
        for group_key, group_features in features.groupby("group", sort=True):
            columns = group_features["feature"].astype(str).tolist()
            deltas = self._permutation_deltas(
                model=model,
                features=target_features,
                actual=actual,
                columns=columns,
                baseline=baseline,
                repeats=repeats,
                rng=rng,
            )
            group_records.append(
                {
                    "group": str(group_key),
                    "source": str(group_features["source"].iloc[0]),
                    "features": len(columns),
                    "importance_log_loss": float(np.mean(deltas)),
                    "importance_std": float(np.std(deltas)),
                }
            )
        groups = self._rank(pd.DataFrame.from_records(group_records), "group")
        return FeatureImportanceResult(
            features=features,
            groups=groups,
            baseline_log_loss=baseline,
            model_name=model_name,
            tournament_year=tournament_year,
            trained_matches=len(history),
            test_matches=len(target),
            repeats=repeats,
            poisson_weight=float(getattr(model, "poisson_weight", float("nan"))),
        )

    @staticmethod
    def classify(feature: str) -> tuple[str, str]:
        """Relaciona cada columna con su familia conceptual y fuente original."""

        if "fifa" in feature:
            return "Ranking FIFA", "Dato-Futbol / FIFA"
        if "elo" in feature:
            return "Elo", "Resultados mundialistas"
        if "h2h" in feature:
            return "Enfrentamientos directos", "Resultados mundialistas"
        if "rest_days" in feature:
            return "Descanso", "Resultados mundialistas"
        if "goals" in feature:
            return "Goles recientes", "Resultados mundialistas"
        if any(token in feature for token in ("win_rate", "draw_rate", "loss_rate", "recent_form")):
            return "Forma y resultados", "Resultados mundialistas"
        if any(
            token in feature for token in ("matches", "knockout_experience", "current_tournament")
        ):
            return "Experiencia", "Resultados mundialistas"
        return "Contexto del partido", "Resultados mundialistas"

    def _permutation_deltas(
        self,
        *,
        model: PredictionModel,
        features: pd.DataFrame,
        actual: pd.Series,
        columns: list[str],
        baseline: float,
        repeats: int,
        rng: np.random.Generator,
    ) -> list[float]:
        deltas: list[float] = []
        original = features[columns].to_numpy(copy=True)
        for _ in range(repeats):
            permuted = features.copy()
            order = rng.permutation(len(features))
            permuted.loc[:, columns] = original[order]
            output = model.predict(permuted)
            deltas.append(self._loss(actual, output) - baseline)
        return deltas

    @classmethod
    def _loss(cls, actual: pd.Series, output: BatchPrediction) -> float:
        probabilities = np.column_stack(
            [output.probability_away, output.probability_draw, output.probability_home]
        )
        probabilities = np.clip(probabilities, 1e-9, 1.0)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return float(log_loss(actual, probabilities, labels=cls.LABELS))

    @staticmethod
    def _rank(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
        ranked = frame.sort_values(
            ["importance_log_loss", label_column], ascending=[False, True]
        ).reset_index(drop=True)
        positive = ranked["importance_log_loss"].clip(lower=0.0)
        denominator = float(positive.sum())
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        ranked["positive_importance_percent"] = (
            positive / denominator * 100.0 if denominator > 0.0 else 0.0
        )
        ranked["interpretation"] = np.where(
            ranked["importance_log_loss"].gt(0.0),
            "Ayuda fuera de muestra",
            "Sin aporte estable detectado",
        )
        return ranked


class FeatureImportanceReport:
    """Presenta familias y variables en CSV, Markdown y SVG."""

    def write(
        self,
        result: FeatureImportanceResult,
        output_directory: Path,
    ) -> FeatureImportanceArtifacts:
        output_directory.mkdir(parents=True, exist_ok=True)
        stem = f"features_{result.model_name}_{result.tournament_year}"
        features_csv = output_directory / f"{stem}.csv"
        groups_csv = output_directory / f"{stem}_groups.csv"
        markdown = output_directory / f"{stem}.md"
        svg = output_directory / f"{stem}.svg"
        result.features.to_csv(features_csv, index=False)
        result.groups.to_csv(groups_csv, index=False)
        markdown.write_text(self._markdown(result), encoding="utf-8")
        svg.write_text(self._svg(result), encoding="utf-8")
        return FeatureImportanceArtifacts(features_csv, groups_csv, markdown, svg)

    @staticmethod
    def _markdown(result: FeatureImportanceResult) -> str:
        lines = [
            f"# Importancia de variables — Mundial {result.tournament_year}",
            "",
            f"Modelo `{result.model_name}` entrenado con {result.trained_matches} "
            "partidos anteriores y "
            f"explicado sobre {result.test_matches} partidos. Log loss base: "
            f"**{result.baseline_log_loss:.4f}**.",
        ]
        if np.isfinite(result.poisson_weight):
            lines.extend(
                [
                    "",
                    f"Peso elegido sin mirar el Mundial objetivo: Poisson "
                    f"**{result.poisson_weight:.0%}** y Keras "
                    f"**{1.0 - result.poisson_weight:.0%}**.",
                ]
            )
        lines.extend(
            [
                "",
                "La importancia es el aumento promedio del log loss al permutar la señal. "
                "Un valor positivo indica que el modelo perdía calidad sin esa información. "
                "Es una asociación post-hoc, no una "
                "relación causal, y no se usó para seleccionar el modelo con el Mundial objetivo.",
                "",
                "## Familias de variables",
                "",
                "| Pos. | Familia | Fuente | Variables | Δ log loss | "
                "Aporte positivo | Interpretación |",
                "|---:|---|---|---:|---:|---:|---|",
            ]
        )
        for _, row in result.groups.iterrows():
            lines.append(
                f"| {int(row['rank'])} | {row['group']} | {row['source']} | "
                f"{int(row['features'])} | {row['importance_log_loss']:.4f} ± "
                f"{row['importance_std']:.4f} | {row['positive_importance_percent']:.1f}% | "
                f"{row['interpretation']} |"
            )
        lines.extend(
            [
                "",
                "## Variables individuales más influyentes",
                "",
                "| Pos. | Variable | Familia | Δ log loss | Aporte positivo |",
                "|---:|---|---|---:|---:|",
            ]
        )
        for _, row in result.features.head(20).iterrows():
            lines.append(
                f"| {int(row['rank'])} | `{row['feature']}` | {row['group']} | "
                f"{row['importance_log_loss']:.4f} ± {row['importance_std']:.4f} | "
                f"{row['positive_importance_percent']:.1f}% |"
            )
        lines.extend(
            [
                "",
                "## Lectura responsable",
                "",
                "- Variables correlacionadas pueden repartirse la importancia y parecer "
                "pequeñas por separado.",
                "- Un valor negativo no prueba que la variable sea dañina; indica que no "
                "mostró aporte estable "
                "en esta edición y con estas permutaciones.",
                "- La familia Ranking FIFA puede quedar en cero cuando no existe una "
                "publicación suficientemente "
                "reciente, como ocurre en 2026 con la copia abierta utilizada.",
                "- La importancia describe el modelo y dataset evaluados, no una ley "
                "general del fútbol.",
            ]
        )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _svg(result: FeatureImportanceResult) -> str:
        groups = result.groups.head(8)
        features = result.features[result.features["importance_log_loss"].gt(0.0)].head(10)
        width = 1200
        group_top = 95
        feature_top = group_top + len(groups) * 36 + 80
        height = feature_top + max(len(features), 1) * 34 + 70
        chart_x, chart_width = 390, 720
        maximum = max(
            float(groups["importance_log_loss"].clip(lower=0.0).max()),
            float(features["importance_log_loss"].max()) if not features.empty else 0.0,
            1e-9,
        )
        title = f"Qué influyó en {result.model_name} · Mundial {result.tournament_year}"
        footer = (
            f"Modelo entrenado con {result.trained_matches} partidos anteriores · "
            f"Log loss base {result.baseline_log_loss:.4f}"
        )
        elements: list[str] = []
        for position, (_, row) in enumerate(groups.iterrows()):
            y = group_top + position * 36
            value = max(float(row["importance_log_loss"]), 0.0)
            bar = chart_width * value / maximum
            color = "#2563eb" if value > 0 else "#cbd5e1"
            elements.append(
                f'<text x="370" y="{y + 18}" text-anchor="end" font-family="system-ui" '
                f'font-size="14" fill="#172033">{escape(str(row["group"]))}</text>'
                f'<rect x="{chart_x}" y="{y}" width="{bar:.1f}" height="24" rx="6" fill="{color}"/>'
                f'<text x="{chart_x + bar + 9:.1f}" y="{y + 17}" font-family="system-ui" '
                f'font-size="13" fill="#657084">{float(row["importance_log_loss"]):.4f}</text>'
            )
        feature_elements: list[str] = []
        for position, (_, row) in enumerate(features.iterrows()):
            y = feature_top + position * 34
            value = float(row["importance_log_loss"])
            bar = chart_width * value / maximum
            feature_elements.append(
                f'<text x="370" y="{y + 17}" text-anchor="end" font-family="ui-monospace" '
                f'font-size="12" fill="#172033">{escape(str(row["feature"]))}</text>'
                f'<rect x="{chart_x}" y="{y}" width="{bar:.1f}" height="22" rx="5" fill="#16a34a"/>'
                f'<text x="{chart_x + bar + 9:.1f}" y="{y + 16}" font-family="system-ui" '
                f'font-size="12" fill="#657084">{value:.4f}</text>'
            )
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" rx="18" fill="#f8fafc"/>
<text x="600" y="34" text-anchor="middle" font-family="system-ui" font-size="23"
font-weight="700" fill="#172033">{escape(title)}</text>
<text x="600" y="61" text-anchor="middle" font-family="system-ui" font-size="14"
fill="#657084">Importancia por permutación: aumento de log loss (más alto = más útil)</text>
<text x="70" y="87" font-family="system-ui" font-size="16" font-weight="700"
fill="#172033">Familias de variables</text>
{"".join(elements)}
<text x="70" y="{feature_top - 25}" font-family="system-ui" font-size="16"
font-weight="700" fill="#172033">Variables individuales con aporte positivo</text>
{"".join(feature_elements)}
<text x="600" y="{height - 24}" text-anchor="middle" font-family="system-ui"
font-size="12" fill="#657084">{escape(footer)}</text>
</svg>
"""
