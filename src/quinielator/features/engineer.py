"""Construcción temporal de variables prepartido."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quinielator.config import FeatureConfig
from quinielator.features.elo import EloRatingCalculator


@dataclass
class _TeamHistory:
    """Estado de una selección antes del siguiente partido."""

    results: deque[float]
    goals_for: deque[int]
    goals_against: deque[int]
    opponents_elo: deque[float]
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    world_cup_knockout_matches: int = 0
    last_match: date | None = None
    tournament_matches: dict[int, int] = field(default_factory=dict)


class TeamFormCalculator:
    """Resume forma reciente y experiencia acumulada."""

    def __init__(self, window: int) -> None:
        self.window = window
        self._history: defaultdict[str, _TeamHistory] = defaultdict(self._new_history)

    def _new_history(self) -> _TeamHistory:
        return _TeamHistory(
            results=deque(maxlen=self.window),
            goals_for=deque(maxlen=self.window),
            goals_against=deque(maxlen=self.window),
            opponents_elo=deque(maxlen=self.window),
        )

    def features(self, team_code: str, tournament_year: int, played_on: date) -> dict[str, float]:
        """Calcula solo con el estado anterior a ``played_on``."""

        history = self._history[team_code]
        weights = np.arange(1, len(history.results) + 1, dtype=float)
        form = float(np.average(history.results, weights=weights)) if len(weights) else 0.5
        goals_for = float(np.mean(history.goals_for)) if history.goals_for else 0.0
        goals_against = float(np.mean(history.goals_against)) if history.goals_against else 0.0
        opponent_elo = float(np.mean(history.opponents_elo)) if history.opponents_elo else 1500.0
        rest_days = 30.0
        if history.last_match is not None:
            rest_days = float(min((played_on - history.last_match).days, 365))
        return {
            "matches": float(history.matches),
            "win_rate": history.wins / history.matches if history.matches else 0.0,
            "draw_rate": history.draws / history.matches if history.matches else 0.0,
            "loss_rate": history.losses / history.matches if history.matches else 0.0,
            "recent_form": form,
            "recent_goals_for": goals_for,
            "recent_goals_against": goals_against,
            "recent_opponent_elo": opponent_elo,
            "knockout_experience": float(history.world_cup_knockout_matches),
            "current_tournament_matches": float(history.tournament_matches.get(tournament_year, 0)),
            "rest_days": rest_days,
        }

    def update(
        self,
        *,
        team_code: str,
        opponent_elo: float,
        goals_for: int,
        goals_against: int,
        tournament_year: int,
        played_on: date,
        knockout: bool,
    ) -> None:
        history = self._history[team_code]
        result = 1.0 if goals_for > goals_against else 0.0 if goals_for < goals_against else 0.5
        history.results.append(result)
        history.goals_for.append(goals_for)
        history.goals_against.append(goals_against)
        history.opponents_elo.append(opponent_elo)
        history.matches += 1
        history.wins += int(result == 1.0)
        history.draws += int(result == 0.5)
        history.losses += int(result == 0.0)
        history.world_cup_knockout_matches += int(knockout)
        history.last_match = played_on
        history.tournament_matches[tournament_year] = (
            history.tournament_matches.get(tournament_year, 0) + 1
        )


class FifaRankingLookup:
    """Busca la última publicación estrictamente anterior al partido."""

    def __init__(self, rankings: pd.DataFrame, max_age_days: int = 365) -> None:
        self.max_age_days = max_age_days
        self._entries: dict[str, tuple[list[date], list[int], list[float]]] = {}
        local = rankings.copy()
        local["published_on"] = pd.to_datetime(local["published_on"]).dt.date
        for code, group in local.groupby("team_code", sort=False):
            ordered = group.sort_values("published_on")
            self._entries[str(code)] = (
                list(ordered["published_on"]),
                [int(value) for value in ordered["rank"]],
                [float(value) for value in ordered["points"]],
            )

    def before(self, team_code: str, played_on: date) -> tuple[float, float, float, float]:
        """Devuelve rank, puntos, disponibilidad y edad sin mirar presente ni futuro."""

        entries = self._entries.get(team_code)
        if entries is None:
            return np.nan, np.nan, 0.0, np.nan
        dates, ranks, points = entries
        index = bisect_left(dates, played_on) - 1
        if index < 0:
            return np.nan, np.nan, 0.0, np.nan
        age_days = float((played_on - dates[index]).days)
        if age_days > self.max_age_days:
            return np.nan, np.nan, 0.0, age_days
        return float(ranks[index]), points[index], 1.0, age_days


class HeadToHeadCalculator:
    """Historial directo expresado desde el orden actual de equipos."""

    def __init__(self) -> None:
        self._points: defaultdict[tuple[str, str], list[float]] = defaultdict(list)

    def features(self, home: str, away: str) -> tuple[float, float]:
        values = self._points[(home, away)]
        return (float(np.mean(values)) if values else 0.5, float(len(values)))

    def update(self, home: str, away: str, home_goals: int, away_goals: int) -> None:
        home_result = 1.0 if home_goals > away_goals else 0.0 if home_goals < away_goals else 0.5
        self._points[(home, away)].append(home_result)
        self._points[(away, home)].append(1.0 - home_result)


class FeatureEngineer:
    """Orquesta Elo, forma, ranking y enfrentamientos en orden temporal."""

    CONFEDERATION_CODES = {
        "UNK": 0.0,
        "AFC": 1.0,
        "CAF": 2.0,
        "CONCACAF": 3.0,
        "CONMEBOL": 4.0,
        "OFC": 5.0,
        "UEFA": 6.0,
    }
    STAGE_CODES = {
        "group": 0.0,
        "round_of_32": 1.0,
        "round_of_16": 2.0,
        "quarterfinal": 3.0,
        "semifinal": 4.0,
        "third_place": 5.0,
        "final": 6.0,
        "other": 0.5,
    }

    def __init__(self, config: FeatureConfig, rankings: pd.DataFrame) -> None:
        self.config = config
        self.elo = EloRatingCalculator(
            config.elo_initial,
            config.elo_k_factor,
            config.elo_home_advantage,
        )
        self.form = TeamFormCalculator(config.rolling_window)
        self.rankings = FifaRankingLookup(rankings)
        self.head_to_head = HeadToHeadCalculator()

    def before_match(self, row: pd.Series) -> dict[str, float]:
        """Crea variables sin actualizar todavía el estado con el resultado."""

        home = str(row["home_team_code"])
        away = str(row["away_team_code"])
        played_on = pd.Timestamp(row["match_date"]).date()
        year = int(row["tournament_year"])
        home_elo = self.elo.rating(home)
        away_elo = self.elo.rating(away)
        home_form = self.form.features(home, year, played_on)
        away_form = self.form.features(away, year, played_on)
        home_rank, home_points, home_rank_available, home_rank_age = self.rankings.before(
            home, played_on
        )
        away_rank, away_points, away_rank_available, away_rank_age = self.rankings.before(
            away, played_on
        )
        h2h_home_points, h2h_matches = self.head_to_head.features(home, away)

        features: dict[str, float] = {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": home_elo - away_elo,
            "elo_expected_home": self.elo.expected_home(home, away),
            "home_fifa_rank": home_rank,
            "away_fifa_rank": away_rank,
            "fifa_rank_difference": away_rank - home_rank,
            "home_fifa_points": home_points,
            "away_fifa_points": away_points,
            "fifa_points_difference": home_points - away_points,
            "fifa_points_relative_difference": (home_points - away_points)
            / max(abs(home_points) + abs(away_points), 1.0),
            "home_fifa_available": home_rank_available,
            "away_fifa_available": away_rank_available,
            "fifa_ranking_available": min(home_rank_available, away_rank_available),
            "home_fifa_age_days": home_rank_age,
            "away_fifa_age_days": away_rank_age,
            "home_fifa_stale": float(home_rank_available == 0.0 and not np.isnan(home_rank_age)),
            "away_fifa_stale": float(away_rank_available == 0.0 and not np.isnan(away_rank_age)),
            "home_is_host": float(row["home_is_host"]),
            "away_is_host": float(row["away_is_host"]),
            "knockout_stage": float(row["knockout_stage"]),
            "stage_code": self.STAGE_CODES.get(str(row["stage"]), 0.5),
            "tournament_year_scaled": (year - 1930.0) / 100.0,
            "same_confederation": float(row["home_confederation"] == row["away_confederation"]),
            "home_confederation_code": self.CONFEDERATION_CODES.get(
                str(row["home_confederation"]), 0.0
            ),
            "away_confederation_code": self.CONFEDERATION_CODES.get(
                str(row["away_confederation"]), 0.0
            ),
            "h2h_home_points": h2h_home_points,
            "h2h_matches": h2h_matches,
        }
        for key, value in home_form.items():
            features[f"home_{key}"] = value
        for key, value in away_form.items():
            features[f"away_{key}"] = value
        for key in home_form:
            features[f"difference_{key}"] = home_form[key] - away_form[key]
        return features

    def observe_match(self, row: pd.Series) -> None:
        """Actualiza estados después de que el resultado se volvió conocido."""

        home = str(row["home_team_code"])
        away = str(row["away_team_code"])
        played_on = pd.Timestamp(row["match_date"]).date()
        year = int(row["tournament_year"])
        home_elo = self.elo.rating(home)
        away_elo = self.elo.rating(away)
        home_goals = int(row["home_goals_90"])
        away_goals = int(row["away_goals_90"])
        knockout = bool(row["knockout_stage"])
        self.form.update(
            team_code=home,
            opponent_elo=away_elo,
            goals_for=home_goals,
            goals_against=away_goals,
            tournament_year=year,
            played_on=played_on,
            knockout=knockout,
        )
        self.form.update(
            team_code=away,
            opponent_elo=home_elo,
            goals_for=away_goals,
            goals_against=home_goals,
            tournament_year=year,
            played_on=played_on,
            knockout=knockout,
        )
        self.head_to_head.update(home, away, home_goals, away_goals)
        self.elo.update(
            home,
            away,
            int(row["home_goals_total"]),
            int(row["away_goals_total"]),
        )


class TemporalDatasetBuilder:
    """Recorre los partidos una vez y produce una tabla auditable."""

    METADATA_COLUMNS = [
        "match_id",
        "tournament_id",
        "tournament_year",
        "match_date",
        "stage",
        "stage_name",
        "home_team_code",
        "home_team_name",
        "away_team_code",
        "away_team_name",
        "home_goals_90",
        "away_goals_90",
        "home_goals_total",
        "away_goals_total",
        "home_penalties",
        "away_penalties",
        "penalty_shootout",
    ]

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    def build(self, matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
        """Construye features antes de observar cada fila y luego actualiza el estado."""

        ordered = matches.sort_values(["match_date", "match_time", "match_id"]).reset_index(
            drop=True
        )
        engineer = FeatureEngineer(self.config, rankings)
        records: list[dict[str, Any]] = []
        for _, row in ordered.iterrows():
            features = engineer.before_match(row)
            record = {column: row[column] for column in self.METADATA_COLUMNS}
            record.update(features)
            home_goals = int(row["home_goals_90"])
            away_goals = int(row["away_goals_90"])
            record["target_outcome"] = (
                "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
            )
            if int(row["home_goals_total"]) > int(row["away_goals_total"]):
                advancing = str(row["home_team_code"])
            elif int(row["home_goals_total"]) < int(row["away_goals_total"]):
                advancing = str(row["away_team_code"])
            elif int(row["home_penalties"]) > int(row["away_penalties"]):
                advancing = str(row["home_team_code"])
            elif int(row["home_penalties"]) < int(row["away_penalties"]):
                advancing = str(row["away_team_code"])
            else:
                advancing = ""
            record["actual_advancing_team"] = advancing
            records.append(record)
            engineer.observe_match(row)
        return pd.DataFrame.from_records(records)

    @staticmethod
    def feature_columns(dataset: pd.DataFrame) -> list[str]:
        """Obtiene solamente variables numéricas prepartido."""

        excluded = set(TemporalDatasetBuilder.METADATA_COLUMNS) | {
            "target_outcome",
            "actual_advancing_team",
            # FIFA cambió varias veces la escala de puntos. Se conservan para
            # auditoría, pero el modelo usa rango y diferencia relativa acotada.
            "home_fifa_points",
            "away_fifa_points",
            "fifa_points_difference",
        }
        return [
            column
            for column in dataset.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(dataset[column])
        ]
