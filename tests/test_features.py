from __future__ import annotations

import pandas as pd
from pandas.testing import assert_series_equal

from quinielator.config import FeatureConfig
from quinielator.features.engineer import FifaRankingLookup, TemporalDatasetBuilder


def test_ranking_lookup_uses_strictly_previous_publication() -> None:
    rankings = pd.DataFrame(
        [
            {"published_on": "2000-01-01", "team_code": "AAA", "rank": 5, "points": 90},
            {"published_on": "2000-02-01", "team_code": "AAA", "rank": 3, "points": 95},
        ]
    )
    lookup = FifaRankingLookup(rankings)
    rank, points, available, age_days = lookup.before("AAA", pd.Timestamp("2000-02-01").date())
    assert (rank, points, available, age_days) == (5.0, 90.0, 1.0, 31.0)


def test_ranking_lookup_rejects_stale_publication() -> None:
    rankings = pd.DataFrame(
        [{"published_on": "2000-01-01", "team_code": "AAA", "rank": 5, "points": 90}]
    )
    lookup = FifaRankingLookup(rankings)
    rank, points, available, age_days = lookup.before("AAA", pd.Timestamp("2002-01-01").date())
    assert pd.isna(rank) and pd.isna(points)
    assert (available, age_days) == (0.0, 731.0)


def test_future_result_cannot_change_past_features(
    synthetic_matches: pd.DataFrame, synthetic_rankings: pd.DataFrame
) -> None:
    builder = TemporalDatasetBuilder(FeatureConfig())
    original = builder.build(synthetic_matches, synthetic_rankings)
    modified_matches = synthetic_matches.copy()
    modified_matches.loc[2, ["home_goals_90", "home_goals_total"]] = [8, 8]
    modified = builder.build(modified_matches, synthetic_rankings)
    feature_columns = builder.feature_columns(original)
    assert_series_equal(
        original.loc[0, feature_columns],
        modified.loc[0, feature_columns],
        check_names=False,
    )
    assert_series_equal(
        original.loc[1, feature_columns],
        modified.loc[1, feature_columns],
        check_names=False,
    )


def test_features_are_pre_match(
    synthetic_matches: pd.DataFrame, synthetic_rankings: pd.DataFrame
) -> None:
    dataset = TemporalDatasetBuilder(FeatureConfig()).build(synthetic_matches, synthetic_rankings)
    assert dataset.loc[0, "home_matches"] == 0
    assert dataset.loc[1, "home_matches"] == 1
    assert dataset.loc[2, "home_matches"] == 1
    assert dataset.loc[2, "fifa_ranking_available"] == 1
