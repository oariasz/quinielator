"""CLI local de Quinielator construida con argparse."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from quinielator.domain import Stage
from quinielator.models import KerasMatchModel, ModelRegistry
from quinielator.presentation import print_predictions
from quinielator.services import QuinielatorApplication


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quinielator",
        description="Laboratorio local de predicciones temporales de Mundiales",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    data = subcommands.add_parser("data", help="Descarga y preparación de datos")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    download = data_commands.add_parser("download", help="Descarga las fuentes con caché")
    download.add_argument("--force", action="store_true", help="Vuelve a descargar")
    data_commands.add_parser("validate", help="Valida las fuentes")
    data_commands.add_parser("prepare", help="Construye features temporales")

    train = subcommands.add_parser("train", help="Entrena y guarda un modelo con toda la historia")
    train.add_argument("--model", default="poisson")

    evaluate = subcommands.add_parser("evaluate", help="Ejecuta backtesting walk-forward")
    evaluate.add_argument("--model", default="poisson")
    evaluate.add_argument("--start-year", type=int)
    evaluate.add_argument("--end-year", type=int)
    evaluate.add_argument(
        "--mode", choices=["strict_editions", "walk_forward"], default="strict_editions"
    )

    report = subcommands.add_parser("report", help="Regenera gráficas e informe")
    report.add_argument("--model", default="poisson")

    interactive = subcommands.add_parser("interactive", help="Predice, pausa y revela")
    interactive.add_argument("--stage", default="final")
    interactive.add_argument("--start-year", type=int)
    interactive.add_argument("--end-year", type=int)
    interactive.add_argument("--model", default="ensemble")
    interactive.add_argument(
        "--mode", choices=["strict_editions", "walk_forward"], default="strict_editions"
    )

    predict = subcommands.add_parser("predict", help="Lista predicciones históricas filtradas")
    predict.add_argument("--world-cup", type=int, required=True)
    predict.add_argument("--stage", default="final")
    predict.add_argument("--team-a")
    predict.add_argument("--team-b")
    predict.add_argument("--model", default="poisson")

    subcommands.add_parser("models", help="Lista modelos y disponibilidad")
    return parser


def _print_historical_prediction(
    args: argparse.Namespace, application: QuinielatorApplication
) -> None:
    predictions, _ = application.load_or_evaluate_predictions(
        model_name=args.model,
        start_year=args.world_cup,
        end_year=args.world_cup,
    )
    selected = predictions[predictions["stage"].eq(Stage.parse(args.stage).value)]
    if args.team_a:
        term = args.team_a.lower()
        selected = selected[
            selected["home_team_name"].str.lower().str.contains(term)
            | selected["away_team_name"].str.lower().str.contains(term)
            | selected["home_team_code"].str.lower().eq(term)
            | selected["away_team_code"].str.lower().eq(term)
        ]
    if args.team_b:
        term = args.team_b.lower()
        selected = selected[
            selected["home_team_name"].str.lower().str.contains(term)
            | selected["away_team_name"].str.lower().str.contains(term)
            | selected["home_team_code"].str.lower().eq(term)
            | selected["away_team_code"].str.lower().eq(term)
        ]
    if selected.empty:
        print("No se encontraron partidos con esos filtros.")
        return
    columns = [
        "match_date",
        "stage_name",
        "home_team_name",
        "predicted_home_goals",
        "predicted_away_goals",
        "away_team_name",
        "probability_home",
        "probability_draw",
        "probability_away",
    ]
    print(selected[columns].to_string(index=False))


def main(argv: Sequence[str] | None = None) -> None:
    """Ejecuta la CLI y presenta errores recuperables sin traceback ruidoso."""

    args = _parser().parse_args(argv)
    try:
        application = QuinielatorApplication()
        if args.command == "data":
            if args.data_command == "download":
                for path in application.download_data(force=args.force):
                    print(f"Disponible: {path}")
            elif args.data_command == "validate":
                report = application.validate_data()
                print(json.dumps(report, indent=2, ensure_ascii=False))
                if not report["valid"]:
                    raise SystemExit(1)
            elif args.data_command == "prepare":
                for path in application.prepare_data():
                    print(f"Generado: {path}")
        elif args.command == "train":
            print(f"Modelo guardado en: {application.train(args.model)}")
        elif args.command == "evaluate":
            _, metrics = application.evaluate(
                model_name=args.model,
                start_year=args.start_year,
                end_year=args.end_year,
                mode=args.mode,
            )
            print(metrics.to_string(index=False))
        elif args.command == "report":
            for path in application.regenerate_report(args.model):
                print(f"Generado: {path}")
        elif args.command == "interactive":
            print_predictions(
                stage=args.stage,
                start_year=args.start_year,
                end_year=args.end_year,
                model_name=args.model,
                evaluation_mode=args.mode,
            )
        elif args.command == "predict":
            _print_historical_prediction(args, application)
        elif args.command == "models":
            names = ModelRegistry(application.config).names
            print("Modelos: " + ", ".join(names))
            print(f"TensorFlow/Keras disponible: {'sí' if KerasMatchModel.available() else 'no'}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
