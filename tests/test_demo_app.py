from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from t2d_aaf.acquisition.adaptive import (
    STOP_INDEX,
    CandidateValueModel,
    FittedAdaptiveValueModels,
    LockedAdaptivePolicy,
    PreparedAdaptiveData,
    load_locked_adaptive_policy,
    state_feature_names,
)
from t2d_aaf.data.cohort import (
    BASE_FEATURES,
    CORE_ACQUIRABLE_FEATURES,
    EXCLUDED_FROM_MAIN_PREDICTORS,
)
from t2d_aaf.demo.service import (
    DemoService,
    deterministic_case_number,
    predictor_columns_for_demo,
)
from t2d_aaf.features.registry import load_feature_cost_design
from t2d_aaf.survival.baseline import PREDICTOR_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "adaptive_value_policy.joblib"


class _DemoBenefitEstimator:
    def __init__(self, candidate_index: int) -> None:
        self.candidate_index = candidate_index

    def predict(self, state: np.ndarray) -> np.ndarray:
        benefit = 0.20 if self.candidate_index == 0 else 0.0
        return np.full(len(state), benefit, dtype=float)


def _demo_service() -> DemoService:
    rows = 2
    frame = pd.DataFrame(
        {
            feature: np.arange(1, rows + 1, dtype=float) + index
            for index, feature in enumerate(PREDICTOR_FEATURES)
        }
    )
    base_values = frame.loc[:, BASE_FEATURES].to_numpy(dtype=float)
    candidate_values = frame.loc[:, CORE_ACQUIRABLE_FEATURES].to_numpy(dtype=float)
    subset_risks = np.tile(np.linspace(0.01, 0.20, 64), (rows, 1))
    prepared = PreparedAdaptiveData(
        base_values=base_values,
        candidate_values=candidate_values,
        base_source_observed=np.ones_like(base_values),
        candidate_source_observed=np.ones_like(candidate_values),
        subset_risks=subset_risks,
    )
    value_models = FittedAdaptiveValueModels(
        models=tuple(
            CandidateValueModel(
                candidate=feature,
                estimator=_DemoBenefitEstimator(index),
                training_summary={},
            )
            for index, feature in enumerate(CORE_ACQUIRABLE_FEATURES)
        ),
        state_feature_names=state_feature_names(),
        lambda_candidates=np.asarray([], dtype=float),
        seed=17,
        alpha=1.0,
    )
    policy = LockedAdaptivePolicy(
        policy_name="test_policy",
        value_models=value_models,
        selected_cost_penalty=0.10,
        retention_target=0.99,
    )
    design = load_feature_cost_design(
        PROJECT_ROOT / "configs" / "features.yaml",
        PROJECT_ROOT / "configs" / "costs.yaml",
    )
    return DemoService(
        predictor_frame=frame,
        prepared=prepared,
        policy=policy,
        costs=design.costs,
    )


def test_demo_predictor_contract_excludes_metadata_targets_and_leakage() -> None:
    columns = predictor_columns_for_demo()

    assert columns == PREDICTOR_FEATURES
    assert set(columns).isdisjoint(EXCLUDED_FROM_MAIN_PREDICTORS)
    assert set(columns) == set(BASE_FEATURES) | set(CORE_ACQUIRABLE_FEATURES)


def test_demo_service_rejects_any_non_predictor_column() -> None:
    service = _demo_service()
    leaked = service.predictor_frame.assign(event=0)

    with pytest.raises(ValueError, match="only the locked predictor columns"):
        DemoService(
            predictor_frame=leaked,
            prepared=service.prepared,
            policy=service.policy,
            costs=service.costs,
        )


def test_demo_reveals_only_acquired_value_and_stops_without_duplicate_action() -> None:
    service = _demo_service()

    initial = service.snapshot(1)
    assert initial.next_action == "fpg"
    assert initial.acquired_values == ()
    assert initial.total_cost == 0.0

    acquired = service.acquire_recommended(initial)
    after_fpg = service.snapshot(1, acquired)
    assert acquired == ("fpg",)
    assert tuple(value.feature for value in after_fpg.acquired_values) == ("fpg",)
    assert after_fpg.next_action is None
    assert after_fpg.total_cost == pytest.approx(1.0)
    assert service.acquire_recommended(after_fpg) == acquired


def test_demo_case_sampling_is_reproducible_and_bounded() -> None:
    first = deterministic_case_number(20260827, 31_775)
    second = deterministic_case_number(20260827, 31_775)

    assert first == second
    assert 1 <= first <= 31_775


def test_locked_adaptive_artifact_loads_for_inference() -> None:
    if not ADAPTIVE_MODEL_PATH.exists():
        pytest.skip("Run scripts/run_adaptive_acquisition.py to create the local artifact.")

    policy = load_locked_adaptive_policy(ADAPTIVE_MODEL_PATH)

    assert policy.selected_cost_penalty == pytest.approx(0.0010131473450483849)
    assert tuple(model.candidate for model in policy.value_models.models) == tuple(
        CORE_ACQUIRABLE_FEATURES
    )
    assert policy.value_models.state_feature_names == state_feature_names()
    assert STOP_INDEX == len(CORE_ACQUIRABLE_FEATURES)


def test_real_demo_service_uses_test_predictors_only() -> None:
    required = (
        PROJECT_ROOT / "data" / "processed" / "primary_cohort.csv",
        PROJECT_ROOT / "models" / "static_subsets.joblib",
        ADAPTIVE_MODEL_PATH,
    )
    if not all(path.exists() for path in required):
        pytest.skip("Generate the local cohort construction, static baseline, and adaptive acquisition artifacts first.")

    service = DemoService.from_project(PROJECT_ROOT)
    snapshot = service.snapshot(1)

    assert service.participant_count == 31_775
    assert tuple(service.predictor_frame.columns) == PREDICTOR_FEATURES
    assert set(service.predictor_frame.columns).isdisjoint(
        EXCLUDED_FROM_MAIN_PREDICTORS
    )
    assert snapshot.case_label == "Test case 00001"
    assert snapshot.acquired_values == ()
    assert 0.0 <= snapshot.current_risk_at_3_years <= 1.0


def test_random_case_button_resets_acquisition_and_widget(monkeypatch) -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    import streamlit as st

    st.cache_resource.clear()
    monkeypatch.setattr(
        DemoService, "from_project", classmethod(lambda cls, root: _demo_service())
    )
    try:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "app/streamlit_app.py"), default_timeout=30
        ).run()
        assert not app.exception
        next(button for button in app.button if button.label.startswith("Acquire ")).click().run()
        assert app.session_state["acquired_features"] == ["fpg"]
        previous_seed = app.session_state["random_seed"]
        next(button for button in app.button if button.label == "New random case").click().run()
        assert not app.exception
        assert app.session_state["random_seed"] == previous_seed + 1
        assert app.session_state["acquired_features"] == []
        assert app.number_input[0].value == app.session_state["case_number"]
    finally:
        st.cache_resource.clear()
