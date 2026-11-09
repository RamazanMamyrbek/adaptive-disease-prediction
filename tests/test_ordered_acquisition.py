from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from t2d_aaf.acquisition.ordered import (
    ACTION_NAMES,
    DEFAULT_RETENTION_TARGET,
    ThresholdSearchPoint,
    acquisition_summary,
    build_train_threshold_candidates,
    load_ordered_model_bundle,
    prefix_costs,
    select_operating_point,
    simulate_ordered_policy,
)
from t2d_aaf.data.cohort import CORE_ACQUIRABLE_FEATURES, EXCLUDED_FROM_MAIN_PREDICTORS
from t2d_aaf.features.registry import load_feature_cost_design


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_CONFIG_PATH = PROJECT_ROOT / "configs" / "features.yaml"
COST_CONFIG_PATH = PROJECT_ROOT / "configs" / "costs.yaml"
STATIC_METRICS_PATH = PROJECT_ROOT / "reports" / "static_baselines_metrics.json"
STATIC_MODEL_PATH = PROJECT_ROOT / "models" / "static_subsets.joblib"
ORDERED_METRICS_PATH = PROJECT_ROOT / "reports" / "ordered_acquisition_metrics.json"


def _point(
    threshold: float,
    mean_tests: float,
    c_index: float,
    brier: float = 0.01,
) -> ThresholdSearchPoint:
    return ThresholdSearchPoint(
        threshold=threshold,
        mean_acquired_tests=mean_tests,
        median_acquired_tests=mean_tests,
        stop_without_labs_percent=0.0,
        acquire_all_percent=0.0,
        validation_c_index=c_index,
        validation_ipcw_brier=brier,
    )


def test_ordered_action_space_has_only_next_and_stop() -> None:
    assert ACTION_NAMES == ("NEXT", "STOP")


def test_ordered_policy_stops_at_first_risk_below_threshold() -> None:
    stage_risks = np.array(
        [
            [0.05, 0.20, 0.30, 0.40],
            [0.20, 0.05, 0.30, 0.40],
            [0.20, 0.30, 0.40, 0.50],
        ]
    )

    simulation = simulate_ordered_policy(stage_risks, threshold=0.10)

    assert np.array_equal(simulation.stop_stage, np.array([0, 1, 3]))
    assert np.allclose(simulation.selected_risk, np.array([0.05, 0.05, 0.50]))


def test_always_stop_and_always_next_are_explicit_endpoints() -> None:
    risks = np.full((4, len(CORE_ACQUIRABLE_FEATURES) + 1), 0.01)

    always_next = simulate_ordered_policy(risks, threshold=0.0)
    always_stop = simulate_ordered_policy(risks, threshold=math.inf)

    assert np.all(always_next.stop_stage == len(CORE_ACQUIRABLE_FEATURES))
    assert np.all(always_stop.stop_stage == 0)


def test_threshold_candidates_depend_only_on_supplied_train_risks() -> None:
    train_risks = np.linspace(0.001, 0.20, 35).reshape(5, 7)

    first = build_train_threshold_candidates(train_risks, quantile_count=7)
    second = build_train_threshold_candidates(train_risks.copy(), quantile_count=7)

    assert np.array_equal(first, second)
    assert first[0] == 0.0
    assert math.isinf(first[-1])


def test_operating_point_is_cheapest_validation_point_meeting_retention() -> None:
    points = (
        _point(0.0, 6.0, 0.90),
        _point(0.01, 2.0, 0.895),
        _point(0.02, 1.0, 0.88),
        _point(math.inf, 0.0, 0.80),
    )

    selected = select_operating_point(points, retention_target=0.90)

    assert selected.target_validation_c_index == pytest.approx(0.89)
    assert selected.point.threshold == pytest.approx(0.01)
    assert selected.point.mean_acquired_tests == pytest.approx(2.0)


def test_missing_requested_result_is_charged_and_counted() -> None:
    design = load_feature_cost_design(FEATURE_CONFIG_PATH, COST_CONFIG_PATH)
    order = tuple(CORE_ACQUIRABLE_FEATURES)
    simulation = simulate_ordered_policy(
        np.array(
            [
                [0.2] * 7,
                [0.01] + [0.2] * 6,
                [0.2, 0.01] + [0.2] * 5,
            ]
        ),
        threshold=0.1,
    )
    frame = pd.DataFrame({feature: [1.0, np.nan, np.nan] for feature in order})

    summary = acquisition_summary(simulation, frame, design.costs, order)

    assert np.array_equal(simulation.stop_stage, np.array([6, 0, 1]))
    assert summary["next_actions_total"] == 7
    assert summary["cost_by_scheme"]["uniform_unit"]["mean"] == pytest.approx(7 / 3)
    fpg = summary["attempted_source_missingness"][0]
    assert fpg["attempted"] == 2
    assert fpg["source_value_missing"] == 1


def test_static_bundle_uses_validation_greedy_prefixes() -> None:
    if not STATIC_MODEL_PATH.exists():
        pytest.skip("Run scripts/run_static_baselines.py to generate the local static baseline models.")
    bundle = load_ordered_model_bundle(STATIC_METRICS_PATH, STATIC_MODEL_PATH)

    assert bundle.order == (
        "fpg",
        "alt",
        "triglycerides",
        "total_cholesterol",
        "ccr",
        "bun",
    )
    assert set(bundle.order) == set(CORE_ACQUIRABLE_FEATURES)


def test_generated_ordered_results_lock_test_and_leakage_contract() -> None:
    if not ORDERED_METRICS_PATH.exists():
        pytest.skip("Run scripts/run_ordered_acquisition.py to generate ordered acquisition results.")
    raw_text = ORDERED_METRICS_PATH.read_text(encoding="utf-8")
    results = json.loads(raw_text)

    assert results["stage"] == 6
    assert results["static_provenance"]["models_fit_split"] == "train"
    assert results["static_provenance"]["preprocessing_fit_split"] == "train"
    assert results["fixed_order"]["patient_specific_order"] is False
    assert results["ordered_policy"]["action_space"] == ["NEXT", "STOP"]
    assert results["ordered_policy"]["free_choice_among_remaining_tests"] is False
    assert results["tuning"]["test_used_for_threshold_selection"] is False
    assert results["tuning"]["test_thresholds_evaluated"] == 1
    assert results["ordered_policy"]["selected_operating_point"][
        "retention_target"
    ] == pytest.approx(DEFAULT_RETENTION_TARGET)
    selected = results["ordered_policy"]["selected_operating_point"]
    assert 0.0 < selected["threshold"] < 1.0
    assert selected["retained_improvement_fraction"] >= DEFAULT_RETENTION_TARGET
    test = results["evaluation"]["selected_policy"]["test"]
    assert test["acquisition"]["mean_acquired_tests"] < len(CORE_ACQUIRABLE_FEATURES)
    assert sum(
        row["participants"] for row in test["acquisition"]["stage_distribution"]
    ) == test["prediction"]["participants"]
    assert results["state_contract"]["outcome_or_followup_available_to_policy"] is False
    assert results["adaptive_acquisition_policy"] is None
    assert results["participant_level_output"] is None
    assert set(results["features"]["acquisition_candidates"]).isdisjoint(
        EXCLUDED_FROM_MAIN_PREDICTORS
    )
    assert "Infinity" not in raw_text
