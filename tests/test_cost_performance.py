from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from t2d_aaf.data.cohort import EXCLUDED_FROM_MAIN_PREDICTORS
from t2d_aaf.evaluation.cost_performance import (
    calibration_bins,
    fixed_horizon_discrimination,
    fixed_horizon_ipcw_outcome,
    paired_bootstrap_evaluation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "reports" / "cost_performance_metrics.json"


def test_fixed_horizon_discrimination_uses_cases_controls_and_excludes_early_censoring() -> None:
    train_duration = np.array([1.0, 2.0, 4.0, 5.0, 6.0])
    train_event = np.array([1, 0, 0, 1, 0])
    duration = np.array([1.0, 2.0, 4.0, 5.0])
    event = np.array([1, 0, 0, 1])
    risk = np.array([0.9, 0.5, 0.2, 0.1])

    outcome = fixed_horizon_ipcw_outcome(
        train_duration, train_event, duration, event, 3.0
    )
    metrics = fixed_horizon_discrimination(outcome, risk)

    assert np.array_equal(outcome.included, np.array([True, False, True, True]))
    assert metrics["horizon_cases"] == 1
    assert metrics["horizon_controls"] == 2
    assert metrics["excluded_censored_before_or_at_horizon"] == 1
    assert metrics["ipcw_cumulative_dynamic_auc"] == pytest.approx(1.0)
    assert metrics["ipcw_average_precision"] == pytest.approx(1.0)


def test_calibration_bins_are_equal_count_and_cover_every_participant() -> None:
    duration = np.linspace(1.0, 12.0, 20)
    event = np.array([1 if index % 4 == 0 else 0 for index in range(20)])
    risk = np.linspace(0.001, 0.2, 20)

    bins = calibration_bins(duration, event, risk, 3.0, bin_count=5)

    assert len(bins) == 5
    assert sum(int(row["participants"]) for row in bins) == 20
    assert {int(row["participants"]) for row in bins} == {4}
    assert all(0.0 <= float(row["km_observed_risk"]) <= 1.0 for row in bins)
    assert all(
        float(left["mean_predicted_risk"]) < float(right["mean_predicted_risk"])
        for left, right in zip(bins, bins[1:])
    )


def test_paired_bootstrap_is_reproducible_and_preserves_cost_difference() -> None:
    train_duration = np.array([1.0, 2.0, 4.0, 5.0, 6.0, 7.0])
    train_event = np.array([1, 1, 0, 0, 1, 0])
    duration = np.array([1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    event = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    method_risks = {
        "lower_cost": np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]),
        "higher_cost": np.array([0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.10, 0.02]),
    }
    method_costs = {
        "lower_cost": np.ones(8),
        "higher_cost": np.full(8, 3.0),
    }
    arguments = (
        train_duration,
        train_event,
        duration,
        event,
        method_risks,
        method_costs,
        3.0,
    )

    first = paired_bootstrap_evaluation(
        *arguments,
        contrasts=(("lower_cost", "higher_cost"),),
        replicates=20,
        seed=17,
    )
    second = paired_bootstrap_evaluation(
        *arguments,
        contrasts=(("lower_cost", "higher_cost"),),
        replicates=20,
        seed=17,
    )

    assert first == second
    contrast = first["contrasts"]["lower_cost_minus_higher_cost"]["mean_cost"]
    assert contrast["point_estimate"] == pytest.approx(-2.0)
    assert contrast["ci95_lower"] == pytest.approx(-2.0)
    assert contrast["ci95_upper"] == pytest.approx(-2.0)


def test_generated_evaluation_results_preserve_selection_and_leakage_contract() -> None:
    if not METRICS_PATH.exists():
        pytest.skip("Run scripts/evaluate_cost_performance.py to generate cost-performance evaluation results.")
    raw_text = METRICS_PATH.read_text(encoding="utf-8")
    results = json.loads(raw_text)

    assert results["stage"] == 8
    assert results["new_model_fitted_in_evaluation"] is False
    assert results["selection_contract"]["selection_split"] == "validation"
    assert results["selection_contract"]["test_used_for_selection"] is False
    assert results["participant_level_output"] is None
    assert set(results["primary_comparison"]) == {
        "base_only",
        "full_features",
        "static_top_k",
        "random_acquisition",
        "importance_greedy",
        "ordered_acquisition",
        "adaptive_acquisition",
    }
    assert set(results["features"]["acquisition_candidates"]).isdisjoint(
        EXCLUDED_FROM_MAIN_PREDICTORS
    )
    for scheme in results["cost_scheme_sensitivity"].values():
        assert scheme["ordered"]["selection_split"] == "validation"
        assert scheme["ordered"]["test_parameters_evaluated"] == 1
        assert scheme["adaptive"]["selection_split"] == "validation"
        assert scheme["adaptive"]["test_parameters_evaluated"] == 1
    assert results["ablations"]["without_cost"]["result"].startswith("all six")
    assert "Infinity" not in raw_text
