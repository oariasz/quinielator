from __future__ import annotations

from pathlib import Path

import pandas as pd

from quinielator.evaluation import WorldCupVisualReport


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "M1",
                "tournament_year": 2026,
                "match_date": "2026-06-01",
                "stage": "group",
                "stage_name": "Matchday 1",
                "home_team_name": "Alpha",
                "away_team_name": "Beta",
                "predicted_home_goals": 2,
                "predicted_away_goals": 0,
                "probability_home": 0.6,
                "probability_draw": 0.25,
                "probability_away": 0.15,
                "predicted_outcome": "H",
                "actual_home_goals": 2,
                "actual_away_goals": 0,
                "actual_home_goals_total": 2,
                "actual_away_goals_total": 0,
                "actual_outcome": "H",
                "penalty_shootout": 0,
            },
            {
                "match_id": "M2",
                "tournament_year": 2026,
                "match_date": "2026-07-01",
                "stage": "round_of_32",
                "stage_name": "Round of 32",
                "home_team_name": "Gamma",
                "away_team_name": "Delta",
                "predicted_home_goals": 1,
                "predicted_away_goals": 1,
                "probability_home": 0.45,
                "probability_draw": 0.3,
                "probability_away": 0.25,
                "predicted_outcome": "H",
                "actual_home_goals": 1,
                "actual_away_goals": 1,
                "actual_home_goals_total": 1,
                "actual_away_goals_total": 1,
                "actual_outcome": "D",
                "penalty_shootout": 1,
            },
        ]
    )


def test_world_cup_report_uses_90_minute_sign_and_colours(tmp_path: Path) -> None:
    report = WorldCupVisualReport()
    table = report.build_table(_predictions(), 2026)

    assert table["predicted_sign"].tolist() == ["1", "1"]
    assert table["actual_sign"].tolist() == ["1", "X"]
    assert table["sign_correct"].tolist() == [True, False]
    assert "no cuentan para el signo" in table.iloc[1]["after_90_note"]

    artifacts = report.write(
        _predictions(), year=2026, model_name="test", output_directory=tmp_path
    )
    html = artifacts.html.read_text(encoding="utf-8")
    markdown = artifacts.markdown.read_text(encoding="utf-8")
    assert 'class="hit exact"' in html
    assert 'class="miss exact-score"' in html
    assert "🟩" in markdown and "🟥" in markdown
