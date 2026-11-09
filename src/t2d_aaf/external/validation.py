from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from t2d_aaf.acquisition.adaptive import (
    FittedAdaptiveValueModels,
    PreparedAdaptiveData,
    load_adaptive_model_bundle,
    prepare_adaptive_data,
    simulate_adaptive_policy,
    subset_to_mask,
)
from t2d_aaf.acquisition.ordered import prefix_costs, simulate_ordered_policy
from t2d_aaf.data.cohort import CORE_ACQUIRABLE_FEATURES
from t2d_aaf.external.nagala import (
    COMPATIBLE_ACQUIRABLE_FEATURES,
    UNAVAILABLE_ACQUIRABLE_FEATURES,
)
from t2d_aaf.features.registry import CostConfig


@dataclass(frozen=True)
class ExternalMethodVectors:
    predicted_risk: NDArray[np.float64]
    participant_cost: NDArray[np.float64]
    acquired_count: NDArray[np.int64]


@dataclass(frozen=True)
class ExternalPredictionRun:
    methods: Mapping[str, ExternalMethodVectors]
    method_metadata: Mapping[str, Mapping[str, Any]]
    policy_metadata: Mapping[str, Any]
    prepared: PreparedAdaptiveData


def load_adaptive_value_models(path: Path) -> FittedAdaptiveValueModels:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != (
        "adaptive_value_policy_v1"
    ):
        raise ValueError("Unsupported or invalid adaptive acquisition value-model artifact.")
    if tuple(artifact.get("candidate_features", ())) != CORE_ACQUIRABLE_FEATURES:
        raise ValueError("adaptive acquisition value-model candidates differ from the primary contract.")
    models = tuple(artifact.get("value_models", ()))
    if len(models) != len(CORE_ACQUIRABLE_FEATURES):
        raise ValueError("adaptive acquisition artifact must contain one model per candidate.")
    return FittedAdaptiveValueModels(
        models=models,
        state_feature_names=tuple(artifact["state_feature_names"]),
        lambda_candidates=np.empty(0, dtype=float),
        seed=int(artifact["training_seed"]),
        alpha=float(artifact["ridge_alpha"]),
    )


def _parameter(value: Any) -> float:
    return math.inf if value is None else float(value)


def _selected_static_subset(
    static_results: Mapping[str, Any],
    evaluation_results: Mapping[str, Any],
    method: str,
) -> tuple[str, ...]:
    target = float(evaluation_results["selection_contract"]["target_validation_c_index"])
    points = static_results["methods"][method]["points"]
    feasible = [
        point
        for point in points
        if float(point["metrics"]["validation"]["harrell_c_index"]) >= target
    ]
    if not feasible:
        raise ValueError(f"No locked {method} point meets the cost-performance evaluation target.")
    selected = min(feasible, key=lambda point: int(point["acquired_count"]))
    return tuple(selected["acquired_features"])


def _mask_costs(costs: CostConfig, scheme: str) -> NDArray[np.float64]:
    values = []
    for mask in range(2 ** len(CORE_ACQUIRABLE_FEATURES)):
        subset = tuple(
            feature
            for index, feature in enumerate(CORE_ACQUIRABLE_FEATURES)
            if mask & (1 << index)
        )
        values.append(costs.acquisition_cost(subset, scheme=scheme))
    return np.asarray(values, dtype=float)


def _fixed_method(
    risk: NDArray[np.float64],
    acquired_count: int,
    cost: float,
) -> ExternalMethodVectors:
    return ExternalMethodVectors(
        predicted_risk=risk,
        participant_cost=np.full(len(risk), cost, dtype=float),
        acquired_count=np.full(len(risk), acquired_count, dtype=np.int64),
    )


def build_external_prediction_run(
    *,
    frame: pd.DataFrame,
    static_metrics_path: Path,
    static_model_path: Path,
    static_results: Mapping[str, Any],
    ordered_results: Mapping[str, Any],
    adaptive_results: Mapping[str, Any],
    evaluation_results: Mapping[str, Any],
    adaptive_model_path: Path,
    costs: CostConfig,
) -> ExternalPredictionRun:

    if evaluation_results["selection_contract"]["test_used_for_selection"] is not False:
        raise ValueError("cost-performance evaluation must confirm validation-only operating-point selection.")
    bundle = load_adaptive_model_bundle(static_metrics_path, static_model_path)
    prepared = prepare_adaptive_data(bundle, frame)
    value_models = load_adaptive_value_models(adaptive_model_path)
    participants = prepared.participants
    full_compatible = tuple(COMPATIBLE_ACQUIRABLE_FEATURES)
    top_k = _selected_static_subset(static_results, evaluation_results, "static_top_k")
    greedy = _selected_static_subset(static_results, evaluation_results, "importance_greedy")
    if top_k != greedy:
        raise ValueError("Observed cost-performance evaluation Top-K and Greedy subsets must coincide.")
    if not set(top_k).issubset(COMPATIBLE_ACQUIRABLE_FEATURES):
        raise ValueError("The locked static subset is not fully available in NAGALA.")

    default_scheme = costs.default_scheme
    compatible_full_cost = costs.acquisition_cost(
        full_compatible, scheme=default_scheme
    )
    static_cost = costs.acquisition_cost(top_k, scheme=default_scheme)
    methods: dict[str, ExternalMethodVectors] = {
        "base_only": _fixed_method(prepared.subset_risks[:, 0], 0, 0.0),
        "compatible_full": _fixed_method(
            prepared.subset_risks[:, subset_to_mask(full_compatible)],
            len(full_compatible),
            compatible_full_cost,
        ),
        "static_top_k": _fixed_method(
            prepared.subset_risks[:, subset_to_mask(top_k)],
            len(top_k),
            static_cost,
        ),
        "importance_greedy": _fixed_method(
            prepared.subset_risks[:, subset_to_mask(greedy)],
            len(greedy),
            static_cost,
        ),
    }

    primary_order = tuple(ordered_results["fixed_order"]["order"])
    compatible_order = tuple(
        feature for feature in primary_order if feature in COMPATIBLE_ACQUIRABLE_FEATURES
    )
    if primary_order[: len(compatible_order)] != compatible_order:
        raise ValueError("Unavailable ordered actions are not a tail of the primary order.")
    ordered_masks = [
        subset_to_mask(compatible_order[:stage])
        for stage in range(len(compatible_order) + 1)
    ]
    ordered_stage_risks = prepared.subset_risks[:, ordered_masks]
    threshold = _parameter(
        ordered_results["ordered_policy"]["selected_operating_point"]["threshold"]
    )
    ordered = simulate_ordered_policy(ordered_stage_risks, threshold)
    ordered_cost = prefix_costs(
        costs, compatible_order, scheme=default_scheme
    )[ordered.stop_stage]
    methods["ordered_compatible"] = ExternalMethodVectors(
        predicted_risk=ordered.selected_risk,
        participant_cost=ordered_cost,
        acquired_count=ordered.stop_stage.astype(np.int64),
    )

    penalty = _parameter(
        adaptive_results["adaptive_policy"]["selected_operating_point"]["cost_penalty"]
    )
    adaptive = simulate_adaptive_policy(
        prepared,
        value_models,
        costs,
        penalty,
        scheme=default_scheme,
        available_candidates=COMPATIBLE_ACQUIRABLE_FEATURES,
    )
    unavailable_mask = sum(
        1 << CORE_ACQUIRABLE_FEATURES.index(feature)
        for feature in UNAVAILABLE_ACQUIRABLE_FEATURES
    )
    if np.any((adaptive.final_masks & unavailable_mask) != 0):
        raise ValueError("External adaptive policy acquired an unavailable laboratory.")
    adaptive_cost = _mask_costs(costs, default_scheme)[adaptive.final_masks]
    methods["adaptive_compatible"] = ExternalMethodVectors(
        predicted_risk=adaptive.selected_risk,
        participant_cost=adaptive_cost,
        acquired_count=adaptive.acquired_count,
    )

    metadata = {
        "base_only": {
            "definition": "primary Base Only Cox model; family history source-missing",
            "acquired_features": [],
        },
        "compatible_full": {
            "definition": "all four NAGALA-compatible laboratory candidates",
            "acquired_features": list(full_compatible),
        },
        "static_top_k": {
            "definition": "locked cost-performance evaluation Static Top-K subset",
            "acquired_features": list(top_k),
        },
        "importance_greedy": {
            "definition": "locked cost-performance evaluation Importance Greedy subset",
            "acquired_features": list(greedy),
        },
        "ordered_compatible": {
            "definition": "locked threshold along the compatible primary order; forced STOP before CCR",
            "acquired_features": "participant-specific compatible prefix",
        },
        "adaptive_compatible": {
            "definition": "locked lambda with only the four compatible actions plus STOP",
            "acquired_features": "participant-specific compatible subset",
        },
    }
    return ExternalPredictionRun(
        methods=methods,
        method_metadata=metadata,
        policy_metadata={
            "primary_order": list(primary_order),
            "compatible_order": list(compatible_order),
            "ordered_threshold": threshold,
            "adaptive_penalty": penalty,
            "available_adaptive_actions": [*COMPATIBLE_ACQUIRABLE_FEATURES, "STOP"],
            "external_outcomes_used_for_selection_or_recalibration": False,
            "participants": participants,
        },
        prepared=prepared,
    )


def acquisition_summary(vectors: ExternalMethodVectors) -> dict[str, float]:
    count = vectors.acquired_count
    cost = vectors.participant_cost
    return {
        "mean_acquired_tests": float(np.mean(count)),
        "median_acquired_tests": float(np.median(count)),
        "p25_acquired_tests": float(np.quantile(count, 0.25)),
        "p75_acquired_tests": float(np.quantile(count, 0.75)),
        "mean_cost": float(np.mean(cost)),
        "median_cost": float(np.median(cost)),
        "p25_cost": float(np.quantile(cost, 0.25)),
        "p75_cost": float(np.quantile(cost, 0.75)),
        "stop_without_labs_percent": float(100.0 * np.mean(count == 0)),
        "acquire_all_compatible_percent": float(
            100.0 * np.mean(count == len(COMPATIBLE_ACQUIRABLE_FEATURES))
        ),
    }
