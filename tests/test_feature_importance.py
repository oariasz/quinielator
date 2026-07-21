from __future__ import annotations

from pathlib import Path

import pandas as pd

from quinielator.config import (
    ApplicationConfig,
    FeatureConfig,
    ProjectPaths,
    SourceConfig,
    TrainingConfig,
)
from quinielator.evaluation import FeatureImportanceAnalyzer


def _config(tmp_path: Path) -> ApplicationConfig:
    return ApplicationConfig(
        seed=42,
        paths=ProjectPaths(tmp_path, tmp_path / "data", tmp_path / "models", tmp_path / "reports"),
        sources=SourceConfig("", "", "", "", "", ""),
        features=FeatureConfig(),
        training=TrainingConfig(),
    )


def test_feature_importance_trains_only_before_target_year(tmp_path: Path) -> None:
    rows = []
    for index, elo in enumerate([400.0, -400.0] * 6):
        year = 2000 if index < 6 else 2004
        home_goals, away_goals = (2, 0) if elo > 0 else (0, 2)
        rows.append(
            {
                "tournament_year": year,
                "match_date": pd.Timestamp(f"{year}-06-{index % 6 + 1:02d}"),
                "home_goals_90": home_goals,
                "away_goals_90": away_goals,
                "target_outcome": "H" if elo > 0 else "A",
                "elo_difference": elo,
                "noise": float(index % 3),
            }
        )
    result = FeatureImportanceAnalyzer(_config(tmp_path)).analyze(
        pd.DataFrame(rows), model_name="baseline", tournament_year=2004, repeats=3
    )

    elo = result.features.loc[result.features["feature"].eq("elo_difference")].iloc[0]
    assert result.trained_matches == 6
    assert result.test_matches == 6
    assert elo["group"] == "Elo"
    assert elo["importance_log_loss"] > 0
