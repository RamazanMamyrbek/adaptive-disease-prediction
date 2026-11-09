from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from t2d_aaf.data.cohort import EXCLUDED_FROM_MAIN_PREDICTORS
from t2d_aaf.evaluation.survival_metrics import (
    harrell_c_index,
    ipcw_brier_score,
    observed_event_risk_at_horizon,
)
from t2d_aaf.survival.baseline import (
    CONTINUOUS_FEATURES,
    PREDICTOR_FEATURES,
    breslow_baseline_cumulative_hazard,
    build_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "reports" / "ordinary_baseline_metrics.json"


def _predictor_frame(rows: int, *, offset: float = 0.0) -> pd.DataFrame:
    data: dict[str, list[float]] = {}
    for index, feature in enumerate(CONTINUOUS_FEATURES):
        data[feature] = [offset + index + row for row in range(rows)]
    data["gender"] = [1.0 if row % 2 == 0 else 2.0 for row in range(rows)]
    data["family_history_diabetes"] = [float(row % 2) for row in range(rows)]
    return pd.DataFrame(data).loc[:, PREDICTOR_FEATURES]


def test_baseline_predictors_exclude_targets_identifiers_and_site() -> None:
    predictors = set(PREDICTOR_FEATURES)

    assert predictors.isdisjoint(EXCLUDED_FROM_MAIN_PREDICTORS)
    assert "participant_id" not in predictors
    assert "site" not in predictors
    assert "event" not in predictors
    assert "duration_years" not in predictors


def test_preprocessor_statistics_are_fitted_from_training_values_only() -> None:
    train = _predictor_frame(3)
    validation = _predictor_frame(2, offset=10_000.0)
    train.loc[1, "fpg"] = np.nan
    preprocessor = build_preprocessor()

    preprocessor.fit(train)
    transformed_validation = preprocessor.transform(validation)

    continuous = preprocessor.named_transformers_["continuous"]
    fpg_index = CONTINUOUS_FEATURES.index("fpg")
    expected_train_median = float(train["fpg"].median())
    assert continuous.named_steps["imputer"].statistics_[fpg_index] == expected_train_median
    assert expected_train_median < 100.0
    assert np.isfinite(transformed_validation).all()


def test_harrell_c_index_recognizes_perfect_and_reversed_rankings() -> None:
    duration = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.ones(4, dtype=int)

    assert harrell_c_index(duration, event, [4.0, 3.0, 2.0, 1.0]) == pytest.approx(1.0)
    assert harrell_c_index(duration, event, [1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.0)
    assert harrell_c_index(duration, event, [1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.5)


def test_ipcw_brier_is_zero_for_perfect_horizon_survival_predictions() -> None:
    duration = np.array([1.0, 2.0, 4.0, 5.0])
    event = np.ones(4, dtype=int)
    predicted_survival_at_three_years = np.array([0.0, 0.0, 1.0, 1.0])

    score = ipcw_brier_score(
        duration,
        event,
        duration,
        event,
        predicted_survival_at_three_years,
        3.0,
    )

    assert score == pytest.approx(0.0)
    assert observed_event_risk_at_horizon(duration, event, 3.0) == pytest.approx(0.5)


def test_breslow_baseline_hazard_is_finite_and_increasing() -> None:
    duration = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 0, 1])
    linear_predictor = np.array([0.5, -0.1, 0.2, -0.3])

    times, cumulative_hazard, risk_center = breslow_baseline_cumulative_hazard(
        duration,
        event,
        linear_predictor,
    )

    assert np.array_equal(times, np.array([1.0, 2.0, 4.0]))
    assert np.isfinite(cumulative_hazard).all()
    assert np.all(np.diff(cumulative_hazard) > 0)
    assert risk_center == pytest.approx(float(linear_predictor.mean()))


def test_generated_baseline_results_preserve_split_and_leakage_contract() -> None:
    if not METRICS_PATH.exists():
        pytest.skip("Run scripts/train_baseline.py to generate baseline modelling results.")

    results = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    predictors = set(results["predictor_features"])
    metrics = results["evaluation"]["metrics"]

    assert results["stage"] == 3
    assert results["model"]["fit_split"] == "train"
    assert results["preprocessing"]["fit_split"] == "train"
    assert predictors.isdisjoint(results["excluded_from_predictors"])
    assert set(metrics) == {"train", "validation", "test"}
    assert results["cost_configuration"] is None
    assert results["acquisition_policy"] is None
    for split in metrics.values():
        assert 0.5 <= split["harrell_c_index"] <= 1.0
        assert 0.0 <= split["ipcw_brier_at_horizon"] <= 1.0
