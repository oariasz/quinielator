from __future__ import annotations

import pandas as pd

from quinielator.data import DataValidator


def test_validator_accepts_consistent_penalty_match(synthetic_matches: pd.DataFrame) -> None:
    report = DataValidator().validate_matches(synthetic_matches)
    assert report.valid


def test_validator_rejects_duplicate_match(synthetic_matches: pd.DataFrame) -> None:
    duplicated = pd.concat([synthetic_matches, synthetic_matches.iloc[[0]]], ignore_index=True)
    report = DataValidator().validate_matches(duplicated)
    assert not report.valid
    assert "duplicados" in report.errors[0]
