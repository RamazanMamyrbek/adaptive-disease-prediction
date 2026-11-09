from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from t2d_aaf.data.cohort import CORE_ACQUIRABLE_FEATURES
from t2d_aaf.evaluation.survival_metrics import (
    harrell_c_index,
    ipcw_brier_score,
    observed_event_risk_at_horizon,
)
from t2d_aaf.features.registry import CostConfig
from t2d_aaf.survival.baseline import (
    EVALUATION_HORIZON_YEARS,
    baseline_hazard_at,
    transform_predictors,
)
from t2d_aaf.survival.static_baselines import (
    FittedStaticSubset,
    canonical_subset,
    prefix_subsets,
)


POLICY_NAME = "ordered_risk_gate_v1"
ORDER_SOURCE_METHOD = "importance_greedy"
DEFAULT_RETENTION_TARGET = 0.99
DEFAULT_THRESHOLD_QUANTILES = 101
ACTION_NAMES: tuple[str, ...] = ("NEXT", "STOP")


@dataclass(frozen=True)
class OrderedModelBundle:
    order: tuple[str, ...]
    preprocessor: Any
    transformed_feature_names: tuple[str, ...]
    models: Mapping[tuple[str, ...], FittedStaticSubset]
    static_results: Mapping[str, Any]


@dataclass(frozen=True)
class OrderedPolicySimulation:
    stop_stage: NDArray[np.int64]
    selected_risk: NDArray[np.float64]


@dataclass(frozen=True)
class ThresholdSearchPoint:
    threshold: float
    mean_acquired_tests: float
    median_acquired_tests: float
    stop_without_labs_percent: float
    acquire_all_percent: float
    validation_c_index: float
    validation_ipcw_brier: float


@dataclass(frozen=True)
class SelectedOperatingPoint:
    point: ThresholdSearchPoint
    retention_target: float
    target_validation_c_index: float
    base_validation_c_index: float
    full_validation_c_index: float
    retained_improvement_fraction: float


def load_ordered_model_bundle(
    static_metrics_path: Path,
    static_model_path: Path,
) -> OrderedModelBundle:

    static_results = json.loads(static_metrics_path.read_text(encoding="utf-8"))
    if static_results.get("stage") != 5:
        raise ValueError("Ordered acquisition requires static baseline metrics.")
    if static_results.get("evaluation", {}).get("test_used_for_selection") is not False:
        raise ValueError("static baseline must confirm that test did not select the fixed order.")
    method = static_results.get("methods", {}).get(ORDER_SOURCE_METHOD)
    if not isinstance(method, dict) or method.get("selection_data") != "validation only":
        raise ValueError("The fixed ordered baseline must use the validation-derived greedy order.")
    order = tuple(method.get("order", ()))
    if len(order) != len(CORE_ACQUIRABLE_FEATURES) or set(order) != set(
        CORE_ACQUIRABLE_FEATURES
    ):
        raise ValueError("static baseline greedy order must contain every core candidate exactly once.")

    artifact = joblib.load(static_model_path)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != (
        "static_subset_models_v1"
    ):
        raise ValueError("Unsupported or invalid static baseline model artifact.")
    transformed_names = tuple(artifact.get("transformed_feature_names", ()))
    models = artifact.get("models")
    if not transformed_names or not isinstance(models, Mapping):
        raise ValueError("static baseline artifact is missing transformed names or subset models.")
    for subset in prefix_subsets(order):
        fitted = models.get(subset)
        if not isinstance(fitted, FittedStaticSubset):
            raise ValueError(f"static baseline artifact is missing ordered prefix model {subset}.")
        if fitted.acquired_features != subset:
            raise ValueError(f"static baseline prefix model metadata differs for {subset}.")

    return OrderedModelBundle(
        order=order,
        preprocessor=artifact["preprocessor"],
        transformed_feature_names=transformed_names,
        models=models,
        static_results=static_results,
    )


def predict_ordered_stage_risks(
    bundle: OrderedModelBundle,
    frame: pd.DataFrame,
    *,
    horizon: float = EVALUATION_HORIZON_YEARS,
) -> NDArray[np.float64]:

    matrix = transform_predictors(bundle.preprocessor, frame)
    index_by_name = {
        feature: index for index, feature in enumerate(bundle.transformed_feature_names)
    }
    stage_risks: list[NDArray[np.float64]] = []
    for subset in prefix_subsets(bundle.order):
        fitted = bundle.models[subset]
        try:
            indices = tuple(index_by_name[name] for name in fitted.selected_feature_names)
        except KeyError as exc:
            raise ValueError(f"Unknown transformed model feature: {exc.args[0]}") from exc
        predictors = matrix[:, indices]
        linear_predictor = predictors @ fitted.coefficients
        baseline_hazard = baseline_hazard_at(
            fitted.baseline_event_times,
            fitted.baseline_cumulative_hazard,
            horizon,
        )
        risk = 1.0 - np.exp(
            -baseline_hazard * np.exp(linear_predictor - fitted.risk_center)
        )
        if not np.isfinite(risk).all() or ((risk < 0.0) | (risk > 1.0)).any():
            raise ValueError(f"Invalid predicted risk for ordered prefix {subset}.")
        stage_risks.append(risk)
    return np.column_stack(stage_risks)


def build_train_threshold_candidates(
    train_stage_risks: NDArray[np.float64],
    *,
    quantile_count: int = DEFAULT_THRESHOLD_QUANTILES,
) -> NDArray[np.float64]:

    risks = np.asarray(train_stage_risks, dtype=float)
    if risks.ndim != 2 or risks.shape[1] < 2:
        raise ValueError("Stage risks must have base plus at least one acquisition stage.")
    if not np.isfinite(risks).all() or ((risks < 0.0) | (risks > 1.0)).any():
        raise ValueError("Stage risks must be finite probabilities.")
    if quantile_count < 3:
        raise ValueError("quantile_count must be at least 3.")
    decision_stage_risks = risks[:, :-1].reshape(-1)
    quantiles = np.quantile(
        decision_stage_risks,
        np.linspace(0.0, 1.0, quantile_count),
    )
    return np.unique(np.concatenate((np.array([0.0]), quantiles, np.array([np.inf]))))


def simulate_ordered_policy(
    stage_risks: NDArray[np.float64],
    threshold: float,
) -> OrderedPolicySimulation:

    risks = np.asarray(stage_risks, dtype=float)
    if risks.ndim != 2 or risks.shape[1] < 2:
        raise ValueError("Stage risks must be a two-dimensional prefix matrix.")
    if not np.isfinite(risks).all() or ((risks < 0.0) | (risks > 1.0)).any():
        raise ValueError("Stage risks must be finite probabilities.")
    if math.isnan(threshold) or threshold < 0.0:
        raise ValueError("Ordered STOP threshold must be non-negative.")

    final_stage = risks.shape[1] - 1
    stop_stage = np.full(len(risks), final_stage, dtype=np.int64)
    active = np.ones(len(risks), dtype=bool)
    for stage in range(final_stage):
        stop_now = active & (risks[:, stage] < threshold)
        stop_stage[stop_now] = stage
        active[stop_now] = False
    selected_risk = risks[np.arange(len(risks)), stop_stage]
    return OrderedPolicySimulation(stop_stage=stop_stage, selected_risk=selected_risk)


def prefix_costs(
    costs: CostConfig,
    order: Sequence[str],
    *,
    scheme: str | None = None,
) -> NDArray[np.float64]:
    return np.asarray(
        [costs.acquisition_cost(order[:stage], scheme=scheme) for stage in range(len(order) + 1)],
        dtype=float,
    )


def evaluate_threshold_grid(
    thresholds: Sequence[float],
    validation_stage_risks: NDArray[np.float64],
    validation_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    costs: CostConfig,
    order: Sequence[str],
    *,
    horizon: float = EVALUATION_HORIZON_YEARS,
) -> tuple[ThresholdSearchPoint, ...]:

    duration = validation_frame["duration_years"].to_numpy(dtype=float)
    event = validation_frame["event"].to_numpy(dtype=np.int64)
    train_duration = train_frame["duration_years"].to_numpy(dtype=float)
    train_event = train_frame["event"].to_numpy(dtype=np.int64)
    primary_prefix_costs = prefix_costs(costs, order)
    points: list[ThresholdSearchPoint] = []
    for threshold in thresholds:
        simulation = simulate_ordered_policy(validation_stage_risks, float(threshold))
        acquired = simulation.stop_stage.astype(float)
        participant_cost = primary_prefix_costs[simulation.stop_stage]
        points.append(
            ThresholdSearchPoint(
                threshold=float(threshold),
                mean_acquired_tests=float(np.mean(acquired)),
                median_acquired_tests=float(np.median(acquired)),
                stop_without_labs_percent=float(100.0 * np.mean(simulation.stop_stage == 0)),
                acquire_all_percent=float(
                    100.0 * np.mean(simulation.stop_stage == len(order))
                ),
                validation_c_index=harrell_c_index(
                    duration,
                    event,
                    simulation.selected_risk,
                ),
                validation_ipcw_brier=ipcw_brier_score(
                    train_duration,
                    train_event,
                    duration,
                    event,
                    1.0 - simulation.selected_risk,
                    horizon,
                ),
            )
        )
        if not np.isclose(float(np.mean(participant_cost)), points[-1].mean_acquired_tests):
            if costs.default_scheme == "uniform_unit":
                raise ValueError("Uniform-unit mean cost must equal mean acquired test count.")
    return tuple(points)


def select_operating_point(
    search_points: Sequence[ThresholdSearchPoint],
    *,
    retention_target: float = DEFAULT_RETENTION_TARGET,
) -> SelectedOperatingPoint:

    if not 0.0 < retention_target <= 1.0:
        raise ValueError("retention_target must lie in (0, 1].")
    if not search_points:
        raise ValueError("Threshold search cannot be empty.")
    always_next = next((point for point in search_points if point.threshold == 0.0), None)
    always_stop = next(
        (point for point in search_points if math.isinf(point.threshold)), None
    )
    if always_next is None or always_stop is None:
        raise ValueError("Threshold search must include always-NEXT and always-STOP endpoints.")
    base_c_index = always_stop.validation_c_index
    full_c_index = always_next.validation_c_index
    if full_c_index <= base_c_index:
        raise ValueError("Full ordered model must improve validation C-index over base.")
    target_c_index = base_c_index + retention_target * (full_c_index - base_c_index)
    feasible = [
        point for point in search_points if point.validation_c_index >= target_c_index
    ]
    if not feasible:
        raise ValueError("No threshold satisfies the validation retention target.")
    selected = min(
        feasible,
        key=lambda point: (
            point.mean_acquired_tests,
            -point.validation_c_index,
            point.validation_ipcw_brier,
            -point.threshold,
        ),
    )
    retained = (selected.validation_c_index - base_c_index) / (
        full_c_index - base_c_index
    )
    return SelectedOperatingPoint(
        point=selected,
        retention_target=retention_target,
        target_validation_c_index=target_c_index,
        base_validation_c_index=base_c_index,
        full_validation_c_index=full_c_index,
        retained_improvement_fraction=float(retained),
    )


def prediction_metrics(
    simulation: OrderedPolicySimulation,
    evaluation_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    *,
    horizon: float = EVALUATION_HORIZON_YEARS,
) -> dict[str, float | int]:
    duration = evaluation_frame["duration_years"].to_numpy(dtype=float)
    event = evaluation_frame["event"].to_numpy(dtype=np.int64)
    train_duration = train_frame["duration_years"].to_numpy(dtype=float)
    train_event = train_frame["event"].to_numpy(dtype=np.int64)
    observed_risk = observed_event_risk_at_horizon(duration, event, horizon)
    predicted_risk = float(np.mean(simulation.selected_risk))
    return {
        "participants": int(len(evaluation_frame)),
        "events_total": int(event.sum()),
        "harrell_c_index": harrell_c_index(duration, event, simulation.selected_risk),
        "ipcw_brier_at_horizon": ipcw_brier_score(
            train_duration,
            train_event,
            duration,
            event,
            1.0 - simulation.selected_risk,
            horizon,
        ),
        "mean_predicted_risk_at_horizon": predicted_risk,
        "km_observed_risk_at_horizon": observed_risk,
        "calibration_in_the_large_difference": predicted_risk - observed_risk,
    }


def acquisition_summary(
    simulation: OrderedPolicySimulation,
    frame: pd.DataFrame,
    costs: CostConfig,
    order: Sequence[str],
) -> dict[str, Any]:

    stages = simulation.stop_stage
    if len(stages) != len(frame):
        raise ValueError("Policy paths and cohort frame must have equal lengths.")
    final_stage = len(order)
    stage_distribution = []
    for stage in range(final_stage + 1):
        count = int(np.sum(stages == stage))
        stage_distribution.append(
            {
                "acquired_count": stage,
                "acquired_features": list(order[:stage]),
                "participants": count,
                "percent": float(100.0 * count / len(stages)),
            }
        )

    missing_attempts = []
    for position, feature in enumerate(order, start=1):
        attempted = stages >= position
        attempted_count = int(attempted.sum())
        source_missing = frame[feature].isna().to_numpy() & attempted
        missing_count = int(source_missing.sum())
        missing_attempts.append(
            {
                "feature": feature,
                "attempted": attempted_count,
                "source_value_available": attempted_count - missing_count,
                "source_value_missing": missing_count,
                "missing_percent_of_attempts": (
                    float(100.0 * missing_count / attempted_count)
                    if attempted_count
                    else 0.0
                ),
            }
        )

    cost_summary = {}
    for scheme_name in costs.schemes:
        per_stage = prefix_costs(costs, order, scheme=scheme_name)
        values = per_stage[stages]
        cost_summary[scheme_name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p25": float(np.quantile(values, 0.25)),
            "p75": float(np.quantile(values, 0.75)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return {
        "mean_acquired_tests": float(np.mean(stages)),
        "median_acquired_tests": float(np.median(stages)),
        "p25_acquired_tests": float(np.quantile(stages, 0.25)),
        "p75_acquired_tests": float(np.quantile(stages, 0.75)),
        "stop_without_labs_percent": float(100.0 * np.mean(stages == 0)),
        "acquire_all_percent": float(100.0 * np.mean(stages == final_stage)),
        "next_actions_total": int(stages.sum()),
        "stop_actions_total": int(len(stages)),
        "stage_distribution": stage_distribution,
        "cost_by_scheme": cost_summary,
        "attempted_source_missingness": missing_attempts,
        "missing_result_rule": (
            "charge the requested action; use the train-fitted imputation value; "
            "retain the participant and continue normal NEXT/STOP evaluation"
        ),
    }
