"""Adaptadores de las dos fuentes externas del proyecto."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from quinielator.data.normalizer import TeamNameNormalizer
from quinielator.data.repository import DataRepository
from quinielator.domain import Stage


class DataSource(ABC):
    """Contrato para convertir una fuente externa a una tabla interna."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Carga y normaliza la fuente."""


class WorldCupResultsSource(DataSource):
    """Adapta partidos y goles de la base Fjelstul."""

    def __init__(self, repository: DataRepository, normalizer: TeamNameNormalizer) -> None:
        self.repository = repository
        self.normalizer = normalizer

    def load(self) -> pd.DataFrame:
        matches = self.repository.read_raw("matches")
        goals = self.repository.read_raw("goals")
        teams = self.repository.read_raw("teams")
        tournaments = self.repository.read_raw("tournaments")

        mens_label = "FIFA Men's World Cup"
        matches = matches[matches["tournament_name"].astype(str).str.contains(mens_label)].copy()
        goals = goals[goals["tournament_name"].astype(str).str.contains(mens_label)].copy()
        tournaments = tournaments[
            tournaments["tournament_name"].astype(str).str.contains(mens_label)
        ].copy()

        confederations = teams.set_index("team_code")["confederation_code"].to_dict()
        hosts = tournaments.set_index("tournament_id")["host_country"].to_dict()

        goals["minute_regulation"] = pd.to_numeric(goals["minute_regulation"], errors="coerce")
        regulation = goals[goals["minute_regulation"].le(90)].copy()
        regulation["side"] = np.where(regulation["home_team"].eq(1), "home", "away")
        regulation_counts = (
            regulation.groupby(["match_id", "side"]).size().unstack(fill_value=0).reset_index()
        )
        if "home" not in regulation_counts:
            regulation_counts["home"] = 0
        if "away" not in regulation_counts:
            regulation_counts["away"] = 0
        regulation_counts = regulation_counts.rename(
            columns={"home": "home_goals_90", "away": "away_goals_90"}
        )
        matches = matches.merge(regulation_counts, on="match_id", how="left")
        matches[["home_goals_90", "away_goals_90"]] = matches[
            ["home_goals_90", "away_goals_90"]
        ].fillna(0)

        matches["match_date"] = pd.to_datetime(matches["match_date"], errors="raise")
        matches["tournament_year"] = matches["tournament_id"].str.extract(r"(\d{4})")[0].astype(int)
        matches["stage"] = matches["stage_name"].map(lambda value: Stage.parse(str(value)).value)
        matches["home_team_name_original"] = matches["home_team_name"]
        matches["away_team_name_original"] = matches["away_team_name"]
        matches["home_team_name"] = matches["home_team_name"].map(self.normalizer.normalize_name)
        matches["away_team_name"] = matches["away_team_name"].map(self.normalizer.normalize_name)
        matches["home_team_code"] = matches["home_team_code"].map(self.normalizer.normalize_code)
        matches["away_team_code"] = matches["away_team_code"].map(self.normalizer.normalize_code)
        matches["home_confederation"] = matches["home_team_code"].map(confederations).fillna("UNK")
        matches["away_confederation"] = matches["away_team_code"].map(confederations).fillna("UNK")
        matches["host_country"] = matches["tournament_id"].map(hosts).fillna("")
        matches["home_is_host"] = matches.apply(
            lambda row: int(
                self.normalizer.key(str(row["home_team_name_original"]))
                in self.normalizer.key(str(row["host_country"]))
            ),
            axis=1,
        )
        matches["away_is_host"] = matches.apply(
            lambda row: int(
                self.normalizer.key(str(row["away_team_name_original"]))
                in self.normalizer.key(str(row["host_country"]))
            ),
            axis=1,
        )
        matches["home_goals_total"] = matches["home_team_score"].astype(int)
        matches["away_goals_total"] = matches["away_team_score"].astype(int)
        matches["home_penalties"] = matches["home_team_score_penalties"].fillna(0).astype(int)
        matches["away_penalties"] = matches["away_team_score_penalties"].fillna(0).astype(int)
        matches["extra_time"] = matches["extra_time"].astype(int)
        matches["penalty_shootout"] = matches["penalty_shootout"].astype(int)

        # La fuente ordena muchos partidos antiguos con el ganador primero. En sede
        # neutral esa orientación codifica el resultado. Se reemplaza por una regla
        # prepartido: anfitrión primero y, en cualquier otro caso, orden por código.
        same_host_status = matches["home_is_host"].eq(matches["away_is_host"])
        swap_sides = (matches["away_is_host"].eq(1) & matches["home_is_host"].eq(0)) | (
            same_host_status
            & matches["home_team_code"].astype(str).gt(matches["away_team_code"].astype(str))
        )
        side_pairs = [
            ("home_team_name", "away_team_name"),
            ("home_team_name_original", "away_team_name_original"),
            ("home_team_code", "away_team_code"),
            ("home_confederation", "away_confederation"),
            ("home_goals_90", "away_goals_90"),
            ("home_goals_total", "away_goals_total"),
            ("home_penalties", "away_penalties"),
            ("home_is_host", "away_is_host"),
        ]
        for home_column, away_column in side_pairs:
            original_home = matches.loc[swap_sides, home_column].copy()
            matches.loc[swap_sides, home_column] = matches.loc[swap_sides, away_column].to_numpy()
            matches.loc[swap_sides, away_column] = original_home.to_numpy()

        columns = [
            "match_id",
            "tournament_id",
            "tournament_name",
            "tournament_year",
            "match_date",
            "match_time",
            "stage",
            "stage_name",
            "knockout_stage",
            "home_team_name",
            "home_team_name_original",
            "home_team_code",
            "home_confederation",
            "away_team_name",
            "away_team_name_original",
            "away_team_code",
            "away_confederation",
            "home_goals_90",
            "away_goals_90",
            "home_goals_total",
            "away_goals_total",
            "extra_time",
            "penalty_shootout",
            "home_penalties",
            "away_penalties",
            "host_country",
            "home_is_host",
            "away_is_host",
        ]
        return (
            matches[columns]
            .sort_values(["match_date", "match_time", "match_id"])
            .reset_index(drop=True)
        )


class WorldCup2026ResultsSource(DataSource):
    """Adapta el suplemento JSON de openfootball para la edición 2026."""

    HOSTS = {"Canada", "Mexico", "USA"}
    HOST_COUNTRY = "Canada, Mexico, United States"
    TEAM_CODES = {
        "Algeria": "ALG",
        "Argentina": "ARG",
        "Australia": "AUS",
        "Austria": "AUT",
        "Belgium": "BEL",
        "Bosnia & Herzegovina": "BIH",
        "Brazil": "BRA",
        "Canada": "CAN",
        "Cape Verde": "CPV",
        "Colombia": "COL",
        "Croatia": "CRO",
        "Curaçao": "CUW",
        "Czech Republic": "CZE",
        "DR Congo": "COD",
        "Ecuador": "ECU",
        "Egypt": "EGY",
        "England": "ENG",
        "France": "FRA",
        "Germany": "GER",
        "Ghana": "GHA",
        "Haiti": "HAI",
        "Iran": "IRN",
        "Iraq": "IRQ",
        "Ivory Coast": "CIV",
        "Japan": "JPN",
        "Jordan": "JOR",
        "Mexico": "MEX",
        "Morocco": "MAR",
        "Netherlands": "NED",
        "New Zealand": "NZL",
        "Norway": "NOR",
        "Panama": "PAN",
        "Paraguay": "PAR",
        "Portugal": "POR",
        "Qatar": "QAT",
        "Saudi Arabia": "KSA",
        "Scotland": "SCO",
        "Senegal": "SEN",
        "South Africa": "RSA",
        "South Korea": "KOR",
        "Spain": "ESP",
        "Sweden": "SWE",
        "Switzerland": "SUI",
        "Tunisia": "TUN",
        "Turkey": "TUR",
        "USA": "USA",
        "Uruguay": "URU",
        "Uzbekistan": "UZB",
    }
    CONFEDERATIONS = {
        "AUS": "AFC",
        "IRN": "AFC",
        "IRQ": "AFC",
        "JPN": "AFC",
        "JOR": "AFC",
        "QAT": "AFC",
        "KSA": "AFC",
        "KOR": "AFC",
        "UZB": "AFC",
        "ALG": "CAF",
        "CPV": "CAF",
        "COD": "CAF",
        "EGY": "CAF",
        "GHA": "CAF",
        "CIV": "CAF",
        "MAR": "CAF",
        "SEN": "CAF",
        "RSA": "CAF",
        "TUN": "CAF",
        "CAN": "CONCACAF",
        "CUW": "CONCACAF",
        "HAI": "CONCACAF",
        "MEX": "CONCACAF",
        "PAN": "CONCACAF",
        "USA": "CONCACAF",
        "ARG": "CONMEBOL",
        "BRA": "CONMEBOL",
        "COL": "CONMEBOL",
        "ECU": "CONMEBOL",
        "PAR": "CONMEBOL",
        "URU": "CONMEBOL",
        "NZL": "OFC",
    }
    STAGE_NAMES = {
        "Round of 32": Stage.ROUND_OF_32,
        "Round of 16": Stage.ROUND_OF_16,
        "Quarter-final": Stage.QUARTERFINAL,
        "Semi-final": Stage.SEMIFINAL,
        "Match for third place": Stage.THIRD_PLACE,
        "Final": Stage.FINAL,
    }

    def __init__(self, repository: DataRepository, normalizer: TeamNameNormalizer) -> None:
        self.repository = repository
        self.normalizer = normalizer

    def load(self) -> pd.DataFrame:
        payload = self.repository.read_raw_json("world_cup_2026")
        raw_matches = payload.get("matches")
        if not isinstance(raw_matches, list):
            raise ValueError("La fuente 2026 no contiene una lista `matches`")

        records = [
            self._match_record(index, raw_match)
            for index, raw_match in enumerate(raw_matches, start=1)
            if isinstance(raw_match, dict)
        ]
        if len(records) != len(raw_matches):
            raise ValueError("La fuente 2026 contiene partidos con formato inválido")
        return pd.DataFrame.from_records(records).sort_values(
            ["match_date", "match_time", "match_id"], ignore_index=True
        )

    def _match_record(self, index: int, raw: dict[str, Any]) -> dict[str, Any]:
        team1 = str(raw["team1"])
        team2 = str(raw["team2"])
        score = raw.get("score")
        if not isinstance(score, dict) or not isinstance(score.get("ft"), list):
            raise ValueError(f"Marcador inválido en el partido 2026 #{index}")
        score_90 = [int(value) for value in score["ft"]]
        score_total = [int(value) for value in score.get("et", score_90)]
        penalties = [int(value) for value in score.get("p", [0, 0])]
        if not (len(score_90) == len(score_total) == len(penalties) == 2):
            raise ValueError(f"Marcador incompleto en el partido 2026 #{index}")

        round_name = str(raw["round"])
        stage = self.STAGE_NAMES.get(round_name, Stage.GROUP)
        home = self._team(team1)
        away = self._team(team2)
        home_is_host = int(team1 in self.HOSTS)
        away_is_host = int(team2 in self.HOSTS)

        # Misma orientación prepartido del histórico: anfitrión primero; en sede
        # neutral, orden por código. Nunca se consulta el marcador para decidir.
        swap = (away_is_host == 1 and home_is_host == 0) or (
            home_is_host == away_is_host and str(home["code"]) > str(away["code"])
        )
        if swap:
            home, away = away, home
            home_is_host, away_is_host = away_is_host, home_is_host
            score_90.reverse()
            score_total.reverse()
            penalties.reverse()

        return {
            "match_id": f"WC-2026-{index:03d}",
            "tournament_id": "WC-2026",
            "tournament_name": "2026 FIFA Men's World Cup",
            "tournament_year": 2026,
            "match_date": pd.to_datetime(str(raw["date"]), errors="raise"),
            "match_time": str(raw.get("time", "00:00")),
            "stage": stage.value,
            "stage_name": round_name,
            "knockout_stage": int(stage is not Stage.GROUP),
            "home_team_name": home["name"],
            "home_team_name_original": home["original_name"],
            "home_team_code": home["code"],
            "home_confederation": home["confederation"],
            "away_team_name": away["name"],
            "away_team_name_original": away["original_name"],
            "away_team_code": away["code"],
            "away_confederation": away["confederation"],
            "home_goals_90": score_90[0],
            "away_goals_90": score_90[1],
            "home_goals_total": score_total[0],
            "away_goals_total": score_total[1],
            "extra_time": int("et" in score),
            "penalty_shootout": int("p" in score),
            "home_penalties": penalties[0],
            "away_penalties": penalties[1],
            "host_country": self.HOST_COUNTRY,
            "home_is_host": home_is_host,
            "away_is_host": away_is_host,
        }

    def _team(self, original_name: str) -> dict[str, str]:
        try:
            source_code = self.TEAM_CODES[original_name]
        except KeyError as error:
            raise ValueError(f"Equipo 2026 sin código conocido: {original_name}") from error
        code = self.normalizer.normalize_code(source_code)
        confederation = self.CONFEDERATIONS.get(source_code, "UEFA")
        return {
            "code": code,
            "name": self.normalizer.normalize_name(original_name),
            "original_name": original_name,
            "confederation": confederation,
        }


class FifaRankingsSource(DataSource):
    """Adapta el histórico extraído del sitio oficial de FIFA."""

    def __init__(self, repository: DataRepository, normalizer: TeamNameNormalizer) -> None:
        self.repository = repository
        self.normalizer = normalizer

    def load(self) -> pd.DataFrame:
        rankings = self.repository.read_raw("rankings").copy()
        rankings["published_on"] = pd.to_datetime(rankings["date"], errors="raise")
        rankings["team_code"] = rankings["team_short"].map(self.normalizer.normalize_code)
        rankings["team_name"] = rankings["team"].map(self.normalizer.normalize_name)
        rankings["points"] = pd.to_numeric(rankings["total_points"], errors="coerce")
        rankings = rankings.dropna(subset=["published_on", "team_code", "points"])
        rankings["rank"] = (
            rankings.groupby("published_on")["points"]
            .rank(method="min", ascending=False)
            .astype(int)
        )
        return rankings[["published_on", "team_code", "team_name", "rank", "points"]].sort_values(
            ["published_on", "rank"]
        )
