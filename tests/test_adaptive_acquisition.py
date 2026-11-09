from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from t2d_aaf.acquisition.adaptive import (
    DEFAULT_RETENTION_TARGET,
    PAD_INDEX,
    STOP_INDEX,
    AdaptivePolicySimulation,
    CandidateValueModel,
    FittedAdaptiveValueModels,
    LambdaSearchPoint,
    PreparedAdaptiveData,
    adaptive_acquisition_summary,
    build_state_matrix,
    load_adaptive_model_bundle,
    mask_to_subset,
    select_adaptive_operating_point,
    simulate_adaptive_policy,
    state_feature_names,
    subset_to_mask,
)
from t2d_aaf.data.cohort import (
    BASE_FEATURES,
    CORE_ACQUIRABLE_FEATURES,
    EXCLUDED_FROM_MAIN_PREDICTORS,
)
from t2d_aaf.features.registry import load_feature_cost_design


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_CONFIG_PATH = PROJECT_ROOT / "configs" / "features.yaml"
COST_CONFIG_PATH = PROJECT_ROOT / "configs" / "costs.yaml"
STATIC_METRICS_PATH = PROJECT_ROOT / "reports" / "static_baselines_metrics.json"
STATIC_MODEL_PATH = PROJECT_ROOT / "models" / "static_subsets.joblib"
ADAPTIVE_METRICS_PATH = PROJECT_ROOT / "reports" / "adaptive_acquisition_metrics.json"


class _BenefitEstimator:
    def __init__(self, candidate_index: int) -> None:
        self.candidate_index = candidate_index

    def predict(self, state: np.ndarray) -> np.ndarray:
        if self.candidate_index == 0:
            return np.where(state[:, 0] > 0.0, 0.20, 0.0)
        if self.candidate_index == 1:
            return np.where(state[:, 0] < 0.0, 0.20, 0.0)
        return np.zeros(len(state))


def _prepared(participants: int = 2) -> PreparedAdaptiveData:
    base = np.zeros((participants, len(BASE_FEATURES)))
    if participants >= 2:
        base[:2, 0] = [1.0, -1.0]
    candidate_values = np.tile(
        np.arange(1, len(CORE_ACQUIRABLE_FEATURES) + 1, dtype=float),
        (participants, 1),
    )
    subset_risks = np.tile(np.linspace(0.01, 0.20, 64), (participants, 1))
    return PreparedAdaptiveData(
        base_values=base,
        candidate_values=candidate_values,
        base_source_observed=np.ones_like(base),
        candidate_source_observed=np.ones_like(candidate_values),
        subset_risks=subset_risks,
    )


def _value_models() -> FittedAdaptiveValueModels:
    return FittedAdaptiveValueModels(
        models=tuple(
            CandidateValueModel(
                candidate=feature,
                estimator=_BenefitEstimator(index),
                training_summary={},
            )
            for index, feature in enumerate(CORE_ACQUIRABLE_FEATURES)
        ),
        state_feature_names=state_feature_names(),
        lambda_candidates=np.array([0.0, 0.1, np.inf]),
        seed=17,
        alpha=1.0,
    )


def _search_point(
    penalty: float,
    mean_cost: float,
    c_index: float,
) -> LambdaSearchPoint:
    return LambdaSearchPoint(
        cost_penalty=penalty,
        mean_acquired_tests=mean_cost,
        median_acquired_tests=mean_cost,
        mean_primary_cost=mean_cost,
        stop_without_labs_percent=0.0,
        acquire_all_percent=0.0,
        unique_paths=2,
        unique_final_subsets=2,
        validation_c_index=c_index,
        validation_ipcw_brier=0.01,
    )


def test_mask_round_trip_covers_all_candidate_subsets() -> None:
    for mask in range(64):
        assert subset_to_mask(mask_to_subset(mask)) == mask


def test_state_hides_unacquired_values_and_keeps_masks_separate() -> None:
    prepared = _prepared(2)
    masks = np.array([1, 0], dtype=np.int64)

    state = build_state_matrix(prepared, masks)

    candidate_value_start = len(BASE_FEATURES)
    acquisition_mask_start = candidate_value_start + len(CORE_ACQUIRABLE_FEATURES)
    source_candidate_start = (
        acquisition_mask_start
        + len(CORE_ACQUIRABLE_FEATURES)
        + len(BASE_FEATURES)
    )
    assert state.shape == (2, len(state_feature_names()))
    assert state[0, candidate_value_start] == 1.0
    assert np.all(state[0, candidate_value_start + 1 : acquisition_mask_start] == 0.0)
    assert np.all(state[1, candidate_value_start:acquisition_mask_start] == 0.0)
    assert np.array_equal(
        state[:, acquisition_mask_start : acquisition_mask_start + 6],
        np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]),
    )
    assert state[0, source_candidate_start] == 1.0
    assert state[1, source_candidate_start] == 0.0


def test_free_choice_policy_selects_different_tests_for_different_states() -> None:
    design = load_feature_cost_design(FEATURE_CONFIG_PATH, COST_CONFIG_PATH)

    simulation = simulate_adaptive_policy(
        _prepared(2),
        _value_models(),
        design.costs,
        cost_penalty=0.10,
    )

    assert np.array_equal(simulation.action_indices[:, 0], np.array([0, 1]))
    assert np.array_equal(simulation.action_indices[:, 1], np.array([STOP_INDEX, STOP_INDEX]))
    assert np.array_equal(simulation.acquired_count, np.array([1, 1]))
    assert simulation.final_masks[0] != simulation.final_masks[1]


def test_adaptive_endpoints_stop_all_or_acquire_each_candidate_once() -> None:
    design = load_feature_cost_design(FEATURE_CONFIG_PATH, COST_CONFIG_PATH)
    prepared = _prepared(2)
    models = _value_models()

    always_stop = simulate_adaptive_policy(prepared, models, design.costs, math.inf)
    always_acquire = simulate_adaptive_policy(prepared, models, design.costs, 0.0)

    assert np.all(always_stop.action_indices[:, 0] == STOP_INDEX)
    assert np.all(always_stop.acquired_count == 0)
    assert np.all(always_acquire.acquired_count == len(CORE_ACQUIRABLE_FEATURES))
    for row in always_acquire.action_indices:
        acquired = row[(row >= 0) & (row < STOP_INDEX)]
        assert len(np.unique(acquired)) == len(CORE_ACQUIRABLE_FEATURES)
        assert row[-1] == STOP_INDEX


def test_adaptive_policy_never_selects_globally_unavailable_candidates() -> None:
    design = load_feature_cost_design(FEATURE_CONFIG_PATH, COST_CONFIG_PATH)

    simulation = simulate_adaptive_policy(
        _prepared(2),
        _value_models(),
        design.costs,
        cost_penalty=0.0,
        available_candidates=(CORE_ACQUIRABLE_FEATURES[0],),
    )

    assert np.all(simulation.acquired_count == 1)
    assert np.all(simulation.action_indices[:, 0] == 0)
    assert np.all(simulation.action_indices[:, 1] == STOP_INDEX)
    assert np.all(simulation.final_masks == 1)


def test_adaptive_selector_uses_cheapest_validation_point_meeting_target() -> None:
    points = (
        _search_point(0.0, 6.0, 0.90),
        _search_point(0.01, 2.0, 0.895),
        _search_point(0.02, 1.0, 0.88),
        _search_point(math.inf, 0.0, 0.80),
    )

    selected = select_adaptive_operating_point(points, retention_target=0.90)

    assert selected.target_validation_c_index == pytest.approx(0.89)
    assert selected.point.cost_penalty == pytest.approx(0.01)
    assert selected.point.mean_primary_cost == pytest.approx(2.0)


def test_adaptive_missing_result_is_charged_and_reported() -> None:
    design = load_feature_cost_design(FEATURE_CONFIG_PATH, COST_CONFIG_PATH)
    actions = np.full((2, 7), PAD_INDEX, dtype=np.int8)
    actions[0, :2] = [0, STOP_INDEX]
    actions[1, :3] = [1, 0, STOP_INDEX]
    simulation = AdaptivePolicySimulation(
        final_masks=np.array([1, 3]),
        action_indices=actions,
        selected_risk=np.array([0.1, 0.2]),
    )
    frame = pd.DataFrame(
        {
            feature: ([np.nan, 1.0] if feature == "fpg" else [1.0, 1.0])
            for feature in CORE_ACQUIRABLE_FEATURES
        }
    )

    summary = adaptive_acquisition_summary(simulation, frame, design.costs)

    assert summary["cost_by_scheme"]["uniform_unit"]["mean"] == pytest.approx(1.5)
    fpg = summary["attempted_source_missingness"][0]
    assert fpg["attempted"] == 2
    assert fpg["source_value_missing"] == 1


def test_static_adaptive_bundle_contains_all_64_models() -> None:
    if not STATIC_MODEL_PATH.exists():
        pytest.skip("Run scripts/run_static_baselines.py to generate static baseline models.")
    bundle = load_adaptive_model_bundle(STATIC_METRICS_PATH, STATIC_MODEL_PATH)

    assert len(bundle.subset_models) == 64


def test_generated_adaptive_results_lock_adaptive_and_test_contract() -> None:
    if not ADAPTIVE_METRICS_PATH.exists():
        pytest.skip("Run scripts/run_adaptive_acquisition.py to generate adaptive acquisition results.")
    raw_text = ADAPTIVE_METRICS_PATH.read_text(encoding="utf-8")
    results = json.loads(raw_text)

    assert results["stage"] == 7
    assert results["static_provenance"]["models_fit_split"] == "train"
    assert results["value_models"]["fit_split"] == "train"
    assert results["value_models"]["target_uses_event_or_followup_directly"] is False
    assert results["adaptive_policy"]["free_choice_among_remaining_tests"] is True
    assert results["adaptive_policy"]["patient_specific_test_order"] is True
    assert set(results["adaptive_policy"]["action_space"]) == {
        *CORE_ACQUIRABLE_FEATURES,
        "STOP",
    }
    assert results["tuning"]["test_used_for_cost_penalty_selection"] is False
    assert results["tuning"]["test_cost_penalties_evaluated"] == 1
    selected = results["adaptive_policy"]["selected_operating_point"]
    assert selected["retained_improvement_fraction"] >= DEFAULT_RETENTION_TARGET
    test = results["evaluation"]["selected_policy"]["test"]
    assert test["acquisition"]["mean_acquired_tests"] < len(CORE_ACQUIRABLE_FEATURES)
    ordered_test = results["evaluation"]["ordered_reference"]["test"]
    assert test["acquisition"]["cost_by_scheme"]["uniform_unit"]["mean"] < (
        ordered_test["acquisition"]["cost_by_scheme"]["uniform_unit"]["mean"]
    )
    assert test["acquisition"]["unique_paths"] > len(CORE_ACQUIRABLE_FEATURES) + 1
    total_acquisitions = sum(
        row["participants"] for row in test["acquisition"]["acquired_by_feature"]
    )
    assert total_acquisitions == pytest.approx(
        test["acquisition"]["mean_acquired_tests"]
        * test["prediction"]["participants"]
    )
    assert results["state_contract"]["unacquired_candidate_values_available_to_policy"] is False
    assert results["state_contract"]["outcome_or_followup_available_to_policy"] is False
    assert set(results["features"]["acquisition_candidates"]).isdisjoint(
        EXCLUDED_FROM_MAIN_PREDICTORS
    )
    assert results["participant_level_output"] is None
    assert "Infinity" not in raw_text
