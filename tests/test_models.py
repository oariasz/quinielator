from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quinielator.models import BaselineModel, PredictionModel
from quinielator.models.base import score_probabilities


def test_score_probabilities_sum_to_one() -> None:
    output = score_probabilities(np.array([1.5]), np.array([1.0]), max_score=8)
    total = output.probability_home[0] + output.probability_draw[0] + output.probability_away[0]
    assert total == pytest.approx(1.0)
    assert output.expected_home_goals[0] == 1.5


def test_baseline_can_be_saved_and_loaded(tmp_path: pytest.TempPathFactory) -> None:
    features = pd.DataFrame({"elo_difference": [0.0, 100.0, -100.0]})
    targets = pd.DataFrame(
        {"home_goals": [1, 2, 0], "away_goals": [1, 0, 2], "outcome": ["D", "H", "A"]}
    )
    model = BaselineModel().fit(features, targets)
    path = tmp_path / "baseline.joblib"  # type: ignore[operator]
    model.save(path)
    loaded = PredictionModel.load(path)
    assert np.allclose(
        model.predict(features).probability_home,
        loaded.predict(features).probability_home,
    )
