from __future__ import annotations

import json
from pathlib import Path

from quinielator.data.normalizer import TeamNameNormalizer
from quinielator.data.repository import DataRepository
from quinielator.data.sources import WorldCup2026ResultsSource


def test_2026_source_separates_regulation_extra_time_and_penalties(tmp_path: Path) -> None:
    repository = DataRepository(tmp_path)
    payload = {
        "name": "World Cup 2026",
        "matches": [
            {
                "round": "Final",
                "date": "2026-07-19",
                "time": "15:00 UTC-4",
                "team1": "Spain",
                "team2": "Argentina",
                "score": {"ft": [0, 0], "et": [1, 0]},
            },
            {
                "round": "Round of 32",
                "date": "2026-07-01",
                "team1": "Mexico",
                "team2": "South Korea",
                "score": {"ft": [1, 1], "et": [1, 1], "p": [4, 3]},
            },
        ],
    }
    repository.raw_path("world_cup_2026").write_text(json.dumps(payload), encoding="utf-8")

    matches = WorldCup2026ResultsSource(repository, TeamNameNormalizer()).load()
    final = matches.loc[matches["stage"].eq("final")].iloc[0]
    shootout = matches.loc[matches["stage"].eq("round_of_32")].iloc[0]

    assert (final["home_goals_90"], final["away_goals_90"]) == (0, 0)
    assert (final["home_goals_total"], final["away_goals_total"]) == (0, 1)
    assert final["away_team_name"] == "Spain"
    assert final["extra_time"] == 1
    assert shootout["penalty_shootout"] == 1
    assert {shootout["home_penalties"], shootout["away_penalties"]} == {3, 4}
