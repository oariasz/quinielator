"""Fixtures sintéticos pequeños; CI nunca descarga datos reales."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_matches() -> pd.DataFrame:
    """Tres partidos con todas las columnas que consume el constructor temporal."""

    return pd.DataFrame(
        [
            {
                "match_id": "M1",
                "tournament_id": "WC-2000",
                "tournament_name": "2000 FIFA Men's World Cup",
                "tournament_year": 2000,
                "match_date": pd.Timestamp("2000-06-01"),
                "match_time": "12:00",
                "stage": "group",
                "stage_name": "group stage",
                "knockout_stage": 0,
                "home_team_name": "Alpha",
                "home_team_code": "ALP",
                "home_confederation": "UEFA",
                "away_team_name": "Beta",
                "away_team_code": "BET",
                "away_confederation": "CONMEBOL",
                "home_goals_90": 2,
                "away_goals_90": 0,
                "home_goals_total": 2,
                "away_goals_total": 0,
                "extra_time": 0,
                "penalty_shootout": 0,
                "home_penalties": 0,
                "away_penalties": 0,
                "host_country": "Alpha",
                "home_is_host": 1,
                "away_is_host": 0,
            },
            {
                "match_id": "M2",
                "tournament_id": "WC-2000",
                "tournament_name": "2000 FIFA Men's World Cup",
                "tournament_year": 2000,
                "match_date": pd.Timestamp("2000-06-05"),
                "match_time": "12:00",
                "stage": "group",
                "stage_name": "group stage",
                "knockout_stage": 0,
                "home_team_name": "Beta",
                "home_team_code": "BET",
                "home_confederation": "CONMEBOL",
                "away_team_name": "Gamma",
                "away_team_code": "GAM",
                "away_confederation": "AFC",
                "home_goals_90": 1,
                "away_goals_90": 1,
                "home_goals_total": 1,
                "away_goals_total": 1,
                "extra_time": 0,
                "penalty_shootout": 0,
                "home_penalties": 0,
                "away_penalties": 0,
                "host_country": "Alpha",
                "home_is_host": 0,
                "away_is_host": 0,
            },
            {
                "match_id": "M3",
                "tournament_id": "WC-2000",
                "tournament_name": "2000 FIFA Men's World Cup",
                "tournament_year": 2000,
                "match_date": pd.Timestamp("2000-06-10"),
                "match_time": "12:00",
                "stage": "final",
                "stage_name": "final",
                "knockout_stage": 1,
                "home_team_name": "Alpha",
                "home_team_code": "ALP",
                "home_confederation": "UEFA",
                "away_team_name": "Gamma",
                "away_team_code": "GAM",
                "away_confederation": "AFC",
                "home_goals_90": 0,
                "away_goals_90": 0,
                "home_goals_total": 1,
                "away_goals_total": 1,
                "extra_time": 1,
                "penalty_shootout": 1,
                "home_penalties": 4,
                "away_penalties": 3,
                "host_country": "Alpha",
                "home_is_host": 1,
                "away_is_host": 0,
            },
        ]
    )


@pytest.fixture
def synthetic_rankings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "published_on": pd.Timestamp("2000-05-01"),
                "team_code": "ALP",
                "team_name": "Alpha",
                "rank": 1,
                "points": 100.0,
            },
            {
                "published_on": pd.Timestamp("2000-05-01"),
                "team_code": "BET",
                "team_name": "Beta",
                "rank": 2,
                "points": 90.0,
            },
            {
                "published_on": pd.Timestamp("2000-05-01"),
                "team_code": "GAM",
                "team_name": "Gamma",
                "rank": 3,
                "points": 80.0,
            },
        ]
    )
