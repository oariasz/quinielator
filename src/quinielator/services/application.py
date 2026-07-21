"""Fachada de aplicación para CLI, pruebas y experiencia interactiva."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quinielator.config import ApplicationConfig
from quinielator.data import (
    DataDownloader,
    DataRepository,
    DataValidator,
    FifaRankingsSource,
    WorldCup2026ResultsSource,
    WorldCupResultsSource,
)
from quinielator.data.normalizer import TeamNameNormalizer
from quinielator.evaluation.reporting import ReportGenerator
from quinielator.evaluation.walk_forward import WorldCupEvaluator
from quinielator.features import TemporalDatasetBuilder
from quinielator.training import ModelTrainer


class QuinielatorApplication:
    """Expone operaciones completas sin mezclar presentación con negocio."""

    WORLD_CUP_LICENSE = "Fjelstul World Cup Database © 2023 Joshua C. Fjelstul, Ph.D.; CC BY-SA 4.0"
    RANKING_LICENSE = (
        "Datos extraídos del sitio oficial FIFA por Dato-Futbol; el repositorio fuente "
        "no declara licencia, por lo que los CSV descargados no se redistribuyen"
    )
    WORLD_CUP_2026_LICENSE = (
        "openfootball/worldcup.json; datos de dominio público según el repositorio fuente"
    )

    def __init__(self, config: ApplicationConfig | None = None) -> None:
        self.config = config or ApplicationConfig.load()
        self.config.ensure_directories()
        self.config.apply_seeds()
        self.repository = DataRepository(self.config.paths.data)
        self.normalizer = TeamNameNormalizer()

    def download_data(self, *, force: bool = False) -> list[Path]:
        """Descarga las fuentes externas necesarias."""

        downloader = DataDownloader(self.config.paths.raw)
        sources = self.config.sources
        specifications = [
            (
                "Partidos de Mundiales",
                sources.world_cup_matches_url,
                self.repository.RAW_FILES["matches"],
                self.WORLD_CUP_LICENSE,
            ),
            (
                "Goles de Mundiales",
                sources.world_cup_goals_url,
                self.repository.RAW_FILES["goals"],
                self.WORLD_CUP_LICENSE,
            ),
            (
                "Selecciones de Mundiales",
                sources.world_cup_teams_url,
                self.repository.RAW_FILES["teams"],
                self.WORLD_CUP_LICENSE,
            ),
            (
                "Ediciones de Mundiales",
                sources.world_cup_tournaments_url,
                self.repository.RAW_FILES["tournaments"],
                self.WORLD_CUP_LICENSE,
            ),
            (
                "Resultados del Mundial 2026",
                sources.world_cup_2026_url,
                self.repository.RAW_FILES["world_cup_2026"],
                self.WORLD_CUP_2026_LICENSE,
            ),
            (
                "Ranking FIFA histórico",
                sources.fifa_rankings_url,
                self.repository.RAW_FILES["rankings"],
                self.RANKING_LICENSE,
            ),
        ]
        return [
            downloader.download(
                name=name,
                url=url,
                filename=filename,
                license_note=license_note,
                force=force,
            )
            for name, url, filename, license_note in specifications
        ]

    def load_sources(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Carga partidos y rankings normalizados."""

        historical = WorldCupResultsSource(self.repository, self.normalizer).load()
        supplement_2026 = WorldCup2026ResultsSource(self.repository, self.normalizer).load()
        # Si la fuente histórica incorpora 2026 en el futuro, no se duplica la edición.
        supplement_2026 = supplement_2026[
            ~supplement_2026["match_id"].isin(historical["match_id"])
            & ~supplement_2026["tournament_year"].isin(historical["tournament_year"])
        ]
        matches = pd.concat([historical, supplement_2026], ignore_index=True).sort_values(
            ["match_date", "match_time", "match_id"], ignore_index=True
        )
        rankings = FifaRankingsSource(self.repository, self.normalizer).load()
        return matches, rankings

    def validate_data(self) -> dict[str, object]:
        """Valida ambas fuentes y guarda un reporte local."""

        matches, rankings = self.load_sources()
        validator = DataValidator()
        match_report = validator.validate_matches(matches)
        ranking_report = validator.validate_rankings(rankings)
        payload: dict[str, object] = {
            "matches": match_report.as_dict(),
            "rankings": ranking_report.as_dict(),
            "valid": match_report.valid and ranking_report.valid,
        }
        self.repository.save_json("validation_report", payload)
        return payload

    def prepare_data(self) -> tuple[Path, Path, Path]:
        """Normaliza, valida y crea el dataset prepartido temporal."""

        matches, rankings = self.load_sources()
        validator = DataValidator()
        match_report = validator.validate_matches(matches)
        ranking_report = validator.validate_rankings(rankings)
        if not match_report.valid or not ranking_report.valid:
            raise ValueError(
                "Los datos no superaron la validación. Ejecuta `data validate` para ver el detalle."
            )
        builder = TemporalDatasetBuilder(self.config.features)
        dataset = builder.build(matches, rankings)
        matches_path = self.repository.save_processed("matches", matches)
        rankings_path = self.repository.save_processed("rankings", rankings)
        dataset_path = self.repository.save_processed("features", dataset)
        return matches_path, rankings_path, dataset_path

    def load_dataset(self) -> pd.DataFrame:
        """Carga features procesadas y restaura sus fechas."""

        dataset = self.repository.read_processed("features")
        dataset["match_date"] = pd.to_datetime(dataset["match_date"])
        return dataset

    def train(self, model_name: str) -> Path:
        return ModelTrainer(self.config).train(self.load_dataset(), model_name)

    def evaluate(
        self,
        *,
        model_name: str,
        start_year: int | None = None,
        end_year: int | None = None,
        mode: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Ejecuta el backtest y genera todos sus artefactos."""

        evaluation_mode = mode or self.config.training.evaluation_mode
        predictions, metrics = WorldCupEvaluator(self.config).evaluate(
            self.load_dataset(),
            model_name=model_name,
            start_year=start_year,
            end_year=end_year,
            mode=evaluation_mode,
        )
        artifact_name = model_name
        if (
            start_year is not None
            or end_year is not None
            or evaluation_mode != self.config.training.evaluation_mode
        ):
            first = str(start_year) if start_year is not None else "first"
            last = str(end_year) if end_year is not None else "last"
            artifact_name = f"{model_name}_{first}_{last}_{evaluation_mode}"
        ReportGenerator(self.config.paths.reports).generate(predictions, metrics, artifact_name)
        return predictions, metrics

    def load_or_evaluate_predictions(
        self,
        *,
        model_name: str,
        start_year: int | None = None,
        end_year: int | None = None,
        mode: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Reutiliza reportes compatibles o los calcula en ese momento."""

        prediction_path = self.config.paths.reports / f"predictions_{model_name}.csv"
        metrics_path = self.config.paths.reports / f"metrics_{model_name}.csv"
        cache_is_compatible = mode in (None, self.config.training.evaluation_mode)
        if prediction_path.exists() and metrics_path.exists() and cache_is_compatible:
            predictions = pd.read_csv(prediction_path)
            metrics = pd.read_csv(metrics_path)
            if start_year is not None:
                predictions = predictions[predictions["tournament_year"].ge(start_year)]
                metrics = metrics[metrics["tournament_year"].ge(start_year)]
            if end_year is not None:
                predictions = predictions[predictions["tournament_year"].le(end_year)]
                metrics = metrics[metrics["tournament_year"].le(end_year)]
            if not predictions.empty:
                return predictions.reset_index(drop=True), metrics.reset_index(drop=True)
        return self.evaluate(
            model_name=model_name,
            start_year=start_year,
            end_year=end_year,
            mode=mode,
        )

    def regenerate_report(self, model_name: str) -> tuple[Path, Path]:
        prediction_path = self.config.paths.reports / f"predictions_{model_name}.csv"
        metrics_path = self.config.paths.reports / f"metrics_{model_name}.csv"
        if not prediction_path.exists() or not metrics_path.exists():
            command = f"evaluate --model {model_name}"
            raise FileNotFoundError(
                f"No hay evaluación para {model_name}. Ejecuta primero `{command}`."
            )
        predictions = pd.read_csv(prediction_path)
        metrics = pd.read_csv(metrics_path)
        generator = ReportGenerator(self.config.paths.reports)
        generated = generator.generate(predictions, metrics, model_name)
        metrics_by_model: dict[str, pd.DataFrame] = {}
        known_models = {"baseline", "poisson", "outcome", "keras", "ensemble"}
        for path in self.config.paths.reports.glob("metrics_*.csv"):
            discovered_name = path.stem.removeprefix("metrics_")
            if discovered_name in known_models:
                metrics_by_model[discovered_name] = pd.read_csv(path)
        if metrics_by_model:
            generator.generate_comparison(metrics_by_model)
        return generated
