from __future__ import annotations

from collections import Counter
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
from sklearn.linear_model import Ridge

from t2d_aaf.data.cohort import BASE_FEATURES, CORE_ACQUIRABLE_FEATURES
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
    enumerate_candidate_subsets,
)


POLICY_NAME = "adaptive_risk_revision_v1"
VALUE_MODEL_TYPE = "Ridge regression"
VALUE_MODEL_ALPHA = 1.0
TRAIN_STATE_SEED = 20260827
DEFAULT_LAMBDA_QUANTILES = 101
DEFAULT_RETENTION_TARGET = 0.99
STOP_INDEX = len(CORE_ACQUIRABLE_FEATURES)
PAD_INDEX = -1
BIT_VALUES = 1 << np.arange(len(CORE_ACQUIRABLE_FEATURES), dtype=np.int64)


@dataclass(frozen=True)
class AdaptiveModelBundle:
    preprocessor: Any
    transformed_feature_names: tuple[str, ...]
    subset_models: Mapping[tuple[str, ...], FittedStaticSubset]
    static_results: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedAdaptiveData:
    base_values: NDArray[np.float64]
    candidate_values: NDArray[np.float64]
    base_source_observed: NDArray[np.float64]
    candidate_source_observed: NDArray[np.float64]
    subset_risks: NDArray[np.float64]

    @property
    def participants(self) -> int:
        return int(self.subset_risks.shape[0])


@dataclass(frozen=True)
class CandidateValueModel:
    candidate: str
    estimator: Ridge
    training_summary: Mapping[str, float | int]


@dataclass(frozen=True)
class FittedAdaptiveValueModels:
    models: tuple[CandidateValueModel, ...]
    state_feature_names: tuple[str, ...]
    lambda_candidates: NDArray[np.float64]
    seed: int
    alpha: float


@dataclass(frozen=True)
class AdaptivePolicySimulation:
    final_masks: NDArray[np.int64]
    action_indices: NDArray[np.int8]
    selected_risk: NDArray[np.float64]

    @property
    def acquired_count(self) -> NDArray[np.int64]:
        return np.sum(
            (self.action_indices >= 0) & (self.action_indices < STOP_INDEX),
            axis=1,
        ).astype(np.int64)


@dataclass(frozen=True)
class AdaptiveActionBatch:
    action_indices: NDArray[np.int8]
    predicted_benefits: NDArray[np.float64]
    utilities: NDArray[np.float64]


@dataclass(frozen=True)
class LockedAdaptivePolicy:
    policy_name: str
    value_models: FittedAdaptiveValueModels
    selected_cost_penalty: float
    retention_target: float


@dataclass(frozen=True)
class LambdaSearchPoint:
    cost_penalty: float
    mean_acquired_tests: float
    median_acquired_tests: float
    mean_primary_cost: float
    stop_without_labs_percent: float
    acquire_all_percent: float
    unique_paths: int
    unique_final_subsets: int
    validation_c_index: float
    validation_ipcw_brier: float


@dataclass(frozen=True)
class SelectedAdaptiveOperatingPoint:
    point: LambdaSearchPoint
    retention_target: float
    target_validation_c_index: float
    base_validation_c_index: float
    full_validation_c_index: float
    retained_improvement_fraction: float


def mask_to_subset(mask: int) -> tuple[str, ...]:
    if mask < 0 or mask >= 2 ** len(CORE_ACQUIRABLE_FEATURES):
        raise ValueError(f"Invalid candidate mask: {mask}")
    return canonical_subset(
        tuple(
            feature
            for index, feature in enumerate(CORE_ACQUIRABLE_FEATURES)
            if mask & (1 << index)
        )
    )


def subset_to_mask(features: Sequence[str]) -> int:
    subset = canonical_subset(features)
    return int(
        sum(1 << CORE_ACQUIRABLE_FEATURES.index(feature) for feature in subset)
    )


def load_adaptive_model_bundle(
    static_metrics_path: Path,
    static_model_path: Path,
) -> AdaptiveModelBundle:

    static_results = json.loads(static_metrics_path.read_text(encoding="utf-8"))
    if static_results.get("stage") != 5:
        raise ValueError("Adaptive acquisition requires static baseline metrics.")
    if static_results.get("evaluation", {}).get("test_used_for_selection") is not False:
        raise ValueError("static baseline must confirm that test did not select static models.")
    artifact = joblib.load(static_model_path)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != (
        "static_subset_models_v1"
    ):
        raise ValueError("Unsupported or invalid static baseline model artifact.")
    transformed_names = tuple(artifact.get("transformed_feature_names", ()))
    models = artifact.get("models")
    if not transformed_names or not isinstance(models, Mapping):
        raise ValueError("static baseline artifact is missing transformed names or subset models.")
    for subset in enumerate_candidate_subsets():
        fitted = models.get(subset)
        if not isinstance(fitted, FittedStaticSubset):
            raise ValueError(f"static baseline artifact is missing subset model {subset}.")
        if fitted.acquired_features != subset:
            raise ValueError(f"static baseline model metadata differs for subset {subset}.")
    return AdaptiveModelBundle(
        preprocessor=artifact["preprocessor"],
        transformed_feature_names=transformed_names,
        subset_models=models,
        static_results=static_results,
    )


def state_feature_names() -> tuple[str, ...]:
    return (
        *(f"value:{feature}" for feature in BASE_FEATURES),
        *(f"value:{feature}" for feature in CORE_ACQUIRABLE_FEATURES),
        *(f"acquired:{feature}" for feature in CORE_ACQUIRABLE_FEATURES),
        *(f"source_observed:{feature}" for feature in BASE_FEATURES),
        *(f"source_observed:{feature}" for feature in CORE_ACQUIRABLE_FEATURES),
        "current_risk_3y",
    )


def prepare_adaptive_data(
    bundle: AdaptiveModelBundle,
    frame: pd.DataFrame,
    *,
    horizon: float = EVALUATION_HORIZON_YEARS,
) -> PreparedAdaptiveData:

    transformed = transform_predictors(bundle.preprocessor, frame)
    index_by_name = {
        feature: index for index, feature in enumerate(bundle.transformed_feature_names)
    }
    try:
        base_indices = tuple(index_by_name[feature] for feature in BASE_FEATURES)
        candidate_indices = tuple(
            index_by_name[feature] for feature in CORE_ACQUIRABLE_FEATURES
        )
    except KeyError as exc:
        raise ValueError(f"Missing transformed adaptive feature: {exc.args[0]}") from exc

    subset_risks = np.empty(
        (len(frame), 2 ** len(CORE_ACQUIRABLE_FEATURES)),
        dtype=float,
    )
    for mask in range(subset_risks.shape[1]):
        subset = mask_to_subset(mask)
        fitted = bundle.subset_models[subset]
        indices = tuple(index_by_name[name] for name in fitted.selected_feature_names)
        linear_predictor = transformed[:, indices] @ fitted.coefficients
        baseline_hazard = baseline_hazard_at(
            fitted.baseline_event_times,
            fitted.baseline_cumulative_hazard,
            horizon,
        )
        subset_risks[:, mask] = 1.0 - np.exp(
            -baseline_hazard * np.exp(linear_predictor - fitted.risk_center)
        )
    if not np.isfinite(subset_risks).all() or (
        (subset_risks < 0.0) | (subset_risks > 1.0)
    ).any():
        raise ValueError("Adaptive subset risks must be finite probabilities.")
    return PreparedAdaptiveData(
        base_values=transformed[:, base_indices],
        candidate_values=transformed[:, candidate_indices],
        base_source_observed=frame.loc[:, BASE_FEATURES].notna().to_numpy(dtype=float),
        candidate_source_observed=(
            frame.loc[:, CORE_ACQUIRABLE_FEATURES].notna().to_numpy(dtype=float)
        ),
        subset_risks=subset_risks,
    )


def _mask_bits(masks: NDArray[np.int64]) -> NDArray[np.float64]:
    return ((masks[:, None] & BIT_VALUES[None, :]) != 0).astype(float)


def build_state_matrix(
    prepared: PreparedAdaptiveData,
    masks: NDArray[np.int64],
    *,
    rows: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:

    participant_rows = np.arange(prepared.participants, dtype=np.int64) if rows is None else rows
    masks_array = np.asarray(masks, dtype=np.int64)
    if len(participant_rows) != len(masks_array):
        raise ValueError("State rows and masks must have equal lengths.")
    if ((masks_array < 0) | (masks_array >= 2 ** len(CORE_ACQUIRABLE_FEATURES))).any():
        raise ValueError("State contains an invalid acquisition mask.")
    acquired = _mask_bits(masks_array)
    current_risk = prepared.subset_risks[participant_rows, masks_array]
    state = np.column_stack(
        (
            prepared.base_values[participant_rows],
            prepared.candidate_values[participant_rows] * acquired,
            acquired,
            prepared.base_source_observed[participant_rows],
            prepared.candidate_source_observed[participant_rows] * acquired,
            current_risk,
        )
    )
    if state.shape[1] != len(state_feature_names()) or not np.isfinite(state).all():
        raise ValueError("Adaptive state matrix is invalid.")
    return state


def _sample_training_masks(
    rng: np.random.Generator,
    participants: int,
    candidate_index: int,
) -> NDArray[np.int64]:
    acquired = rng.integers(
        0,
        2,
        size=(participants, len(CORE_ACQUIRABLE_FEATURES)),
        dtype=np.int8,
    )
    acquired[:, candidate_index] = 0
    return np.sum(acquired * BIT_VALUES[None, :], axis=1, dtype=np.int64)


def fit_adaptive_value_models(
    train_data: PreparedAdaptiveData,
    costs: CostConfig,
    *,
    seed: int = TRAIN_STATE_SEED,
    alpha: float = VALUE_MODEL_ALPHA,
    lambda_quantiles: int = DEFAULT_LAMBDA_QUANTILES,
) -> FittedAdaptiveValueModels:

    if alpha < 0.0 or not math.isfinite(alpha):
        raise ValueError("Ridge alpha must be finite and non-negative.")
    if lambda_quantiles < 3:
        raise ValueError("lambda_quantiles must be at least 3.")
    rng = np.random.default_rng(seed)
    rows = np.arange(train_data.participants, dtype=np.int64)
    fitted: list[CandidateValueModel] = []
    predicted_benefit_per_cost: list[NDArray[np.float64]] = []
    primary_costs = costs.scheme().costs

    for candidate_index, candidate in enumerate(CORE_ACQUIRABLE_FEATURES):
        masks = _sample_training_masks(
            rng,
            train_data.participants,
            candidate_index,
        )
        next_masks = masks | int(BIT_VALUES[candidate_index])
        current_risk = train_data.subset_risks[rows, masks]
        next_risk = train_data.subset_risks[rows, next_masks]
        target = np.abs(next_risk - current_risk)
        state = build_state_matrix(train_data, masks)
        estimator = Ridge(alpha=alpha)
        estimator.fit(state, target)
        predicted = np.clip(estimator.predict(state), 0.0, 1.0)
        predicted_benefit_per_cost.append(predicted / float(primary_costs[candidate]))
        fitted.append(
            CandidateValueModel(
                candidate=candidate,
                estimator=estimator,
                training_summary={
                    "participants": train_data.participants,
                    "sampled_states": int(len(np.unique(masks))),
                    "target_mean_absolute_risk_revision": float(np.mean(target)),
                    "target_median_absolute_risk_revision": float(np.median(target)),
                    "target_p95_absolute_risk_revision": float(np.quantile(target, 0.95)),
                    "training_r2": float(estimator.score(state, target)),
                    "clipped_prediction_mean": float(np.mean(predicted)),
                    "clipped_prediction_max": float(np.max(predicted)),
                },
            )
        )

    ratios = np.concatenate(predicted_benefit_per_cost)
    candidates = np.unique(
        np.concatenate(
            (
                np.array([0.0]),
                np.quantile(ratios, np.linspace(0.0, 1.0, lambda_quantiles)),
                np.array([np.inf]),
            )
        )
    )
    return FittedAdaptiveValueModels(
        models=tuple(fitted),
        state_feature_names=state_feature_names(),
        lambda_candidates=candidates,
        seed=seed,
        alpha=alpha,
    )


def build_cost_penalty_candidates(
    train_data: PreparedAdaptiveData,
    value_models: FittedAdaptiveValueModels,
    costs: CostConfig,
    *,
    scheme: str | None = None,
    lambda_quantiles: int = DEFAULT_LAMBDA_QUANTILES,
) -> NDArray[np.float64]:

    if lambda_quantiles < 3:
        raise ValueError("lambda_quantiles must be at least 3.")
    if tuple(model.candidate for model in value_models.models) != tuple(
        CORE_ACQUIRABLE_FEATURES
    ):
        raise ValueError("Value models must follow the configured candidate order.")
    selected_costs = costs.scheme(scheme).costs
    rng = np.random.default_rng(value_models.seed)
    ratios: list[NDArray[np.float64]] = []
    for candidate_index, model in enumerate(value_models.models):
        masks = _sample_training_masks(
            rng,
            train_data.participants,
            candidate_index,
        )
        state = build_state_matrix(train_data, masks)
        predicted = np.clip(model.estimator.predict(state), 0.0, 1.0)
        ratios.append(predicted / float(selected_costs[model.candidate]))
    values = np.concatenate(ratios)
    return np.unique(
        np.concatenate(
            (
                np.array([0.0]),
                np.quantile(values, np.linspace(0.0, 1.0, lambda_quantiles)),
                np.array([np.inf]),
            )
        )
    )


def load_locked_adaptive_policy(path: Path) -> LockedAdaptivePolicy:

    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != (
        "adaptive_value_policy_v1"
    ):
        raise ValueError("Unsupported or invalid adaptive acquisition policy artifact.")
    if tuple(artifact.get("candidate_features", ())) != tuple(
        CORE_ACQUIRABLE_FEATURES
    ):
        raise ValueError("adaptive acquisition artifact candidate order is incompatible.")
    feature_names = tuple(artifact.get("state_feature_names", ()))
    if feature_names != state_feature_names():
        raise ValueError("adaptive acquisition artifact state contract is incompatible.")
    models = artifact.get("value_models")
    if not isinstance(models, tuple) or len(models) != len(
        CORE_ACQUIRABLE_FEATURES
    ):
        raise ValueError("adaptive acquisition artifact is missing candidate value models.")
    for candidate, model in zip(CORE_ACQUIRABLE_FEATURES, models, strict=True):
        if not isinstance(model, CandidateValueModel) or model.candidate != candidate:
            raise ValueError("adaptive acquisition artifact value models are incompatible.")
    cost_penalty = float(artifact.get("selected_cost_penalty", float("nan")))
    retention_target = float(artifact.get("retention_target", float("nan")))
    if not math.isfinite(cost_penalty) or cost_penalty < 0.0:
        raise ValueError("adaptive acquisition artifact has an invalid selected cost penalty.")
    if not math.isfinite(retention_target) or not 0.0 < retention_target <= 1.0:
        raise ValueError("adaptive acquisition artifact has an invalid retention target.")
    seed = artifact.get("training_seed")
    alpha = artifact.get("ridge_alpha")
    if not isinstance(seed, int) or not isinstance(alpha, (int, float)):
        raise ValueError("adaptive acquisition artifact is missing training metadata.")
    return LockedAdaptivePolicy(
        policy_name=str(artifact.get("policy_name", "")),
        value_models=FittedAdaptiveValueModels(
            models=models,
            state_feature_names=feature_names,
            lambda_candidates=np.asarray([], dtype=float),
            seed=seed,
            alpha=float(alpha),
        ),
        selected_cost_penalty=cost_penalty,
        retention_target=retention_target,
    )


def recommend_adaptive_actions(
    prepared: PreparedAdaptiveData,
    value_models: FittedAdaptiveValueModels,
    costs: CostConfig,
    cost_penalty: float,
    masks: NDArray[np.int64],
    *,
    rows: NDArray[np.int64] | None = None,
    scheme: str | None = None,
    available_candidates: Sequence[str] | None = None,
) -> AdaptiveActionBatch:

    if math.isnan(cost_penalty) or cost_penalty < 0.0:
        raise ValueError("Adaptive cost penalty must be non-negative.")
    if tuple(model.candidate for model in value_models.models) != tuple(
        CORE_ACQUIRABLE_FEATURES
    ):
        raise ValueError("Value models must follow the configured candidate order.")
    masks_array = np.asarray(masks, dtype=np.int64)
    participant_rows = (
        np.arange(len(masks_array), dtype=np.int64)
        if rows is None
        else np.asarray(rows, dtype=np.int64)
    )
    if len(participant_rows) != len(masks_array):
        raise ValueError("Recommendation rows and masks must have equal lengths.")
    selected_scheme = costs.scheme(scheme)
    available = (
        set(CORE_ACQUIRABLE_FEATURES)
        if available_candidates is None
        else set(available_candidates)
    )
    unknown_available = sorted(available - set(CORE_ACQUIRABLE_FEATURES))
    if unknown_available:
        raise ValueError(f"Unknown available candidates: {unknown_available}")

    state = build_state_matrix(
        prepared,
        masks_array,
        rows=participant_rows,
    )
    predicted_benefits = np.column_stack(
        [
            np.clip(model.estimator.predict(state), 0.0, 1.0)
            for model in value_models.models
        ]
    )
    candidate_costs = np.asarray(
        [selected_scheme.costs[feature] for feature in CORE_ACQUIRABLE_FEATURES],
        dtype=float,
    )
    if math.isinf(cost_penalty):
        utilities = np.full_like(predicted_benefits, -np.inf)
    else:
        utilities = predicted_benefits - cost_penalty * candidate_costs[None, :]
    available_mask = np.asarray(
        [feature in available for feature in CORE_ACQUIRABLE_FEATURES],
        dtype=bool,
    )
    utilities[:, ~available_mask] = -np.inf
    utilities[_mask_bits(masks_array).astype(bool)] = -np.inf
    best_candidate = np.argmax(utilities, axis=1)
    best_utility = utilities[np.arange(len(masks_array)), best_candidate]
    action_indices = np.where(
        best_utility >= 0.0,
        best_candidate,
        STOP_INDEX,
    ).astype(np.int8)
    return AdaptiveActionBatch(
        action_indices=action_indices,
        predicted_benefits=predicted_benefits,
        utilities=utilities,
    )


def simulate_adaptive_policy(
    prepared: PreparedAdaptiveData,
    value_models: FittedAdaptiveValueModels,
    costs: CostConfig,
    cost_penalty: float,
    *,
    scheme: str | None = None,
    available_candidates: Sequence[str] | None = None,
) -> AdaptivePolicySimulation:

    participants = prepared.participants
    masks = np.zeros(participants, dtype=np.int64)
    actions = np.full(
        (participants, len(CORE_ACQUIRABLE_FEATURES) + 1),
        PAD_INDEX,
        dtype=np.int8,
    )
    active = np.ones(participants, dtype=bool)

    for stage in range(len(CORE_ACQUIRABLE_FEATURES)):
        rows = np.flatnonzero(active).astype(np.int64)
        if not len(rows):
            break
        recommendation = recommend_adaptive_actions(
            prepared,
            value_models,
            costs,
            cost_penalty,
            masks[rows],
            rows=rows,
            scheme=scheme,
            available_candidates=available_candidates,
        )
        acquire = recommendation.action_indices != STOP_INDEX
        stopped_rows = rows[~acquire]
        actions[stopped_rows, stage] = STOP_INDEX
        active[stopped_rows] = False

        acquired_rows = rows[acquire]
        chosen = recommendation.action_indices[acquire]
        actions[acquired_rows, stage] = chosen.astype(np.int8)
        masks[acquired_rows] |= BIT_VALUES[chosen]

    forced_stop_rows = np.flatnonzero(active).astype(np.int64)
    actions[forced_stop_rows, len(CORE_ACQUIRABLE_FEATURES)] = STOP_INDEX
    selected_risk = prepared.subset_risks[np.arange(participants), masks]
    return AdaptivePolicySimulation(
        final_masks=masks,
        action_indices=actions,
        selected_risk=selected_risk,
    )


def _cost_by_mask(
    costs: CostConfig,
    *,
    scheme: str | None = None,
) -> NDArray[np.float64]:
    return np.asarray(
        [costs.acquisition_cost(mask_to_subset(mask), scheme=scheme) for mask in range(64)],
        dtype=float,
    )


def _path_tuple(action_row: NDArray[np.int8]) -> tuple[int, ...]:
    path: list[int] = []
    for action in action_row:
        if action == PAD_INDEX:
            break
        path.append(int(action))
        if action == STOP_INDEX:
            break
    return tuple(path)


def evaluate_lambda_grid(
    penalties: Sequence[float],
    validation_data: PreparedAdaptiveData,
    validation_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    value_models: FittedAdaptiveValueModels,
    costs: CostConfig,
    *,
    horizon: float = EVALUATION_HORIZON_YEARS,
) -> tuple[LambdaSearchPoint, ...]:

    duration = validation_frame["duration_years"].to_numpy(dtype=float)
    event = validation_frame["event"].to_numpy(dtype=np.int64)
    train_duration = train_frame["duration_years"].to_numpy(dtype=float)
    train_event = train_frame["event"].to_numpy(dtype=np.int64)
    primary_costs = _cost_by_mask(costs)
    points: list[LambdaSearchPoint] = []
    for penalty in penalties:
        simulation = simulate_adaptive_policy(
            validation_data,
            value_models,
            costs,
            float(penalty),
        )
        acquired_count = simulation.acquired_count
        participant_cost = primary_costs[simulation.final_masks]
        points.append(
            LambdaSearchPoint(
                cost_penalty=float(penalty),
                mean_acquired_tests=float(np.mean(acquired_count)),
                median_acquired_tests=float(np.median(acquired_count)),
                mean_primary_cost=float(np.mean(participant_cost)),
                stop_without_labs_percent=float(100.0 * np.mean(acquired_count == 0)),
                acquire_all_percent=float(
                    100.0 * np.mean(acquired_count == len(CORE_ACQUIRABLE_FEATURES))
                ),
                unique_paths=len({_path_tuple(row) for row in simulation.action_indices}),
                unique_final_subsets=int(len(np.unique(simulation.final_masks))),
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
    return tuple(points)


def select_adaptive_operating_point(
    search_points: Sequence[LambdaSearchPoint],
    *,
    retention_target: float = DEFAULT_RETENTION_TARGET,
) -> SelectedAdaptiveOperatingPoint:

    if not 0.0 < retention_target <= 1.0:
        raise ValueError("retention_target must lie in (0, 1].")
    always_acquire = next(
        (point for point in search_points if point.cost_penalty == 0.0), None
    )
    always_stop = next(
        (point for point in search_points if math.isinf(point.cost_penalty)), None
    )
    if always_acquire is None or always_stop is None:
        raise ValueError("Lambda search must include always-acquire and always-STOP endpoints.")
    base_c_index = always_stop.validation_c_index
    full_c_index = always_acquire.validation_c_index
    if full_c_index <= base_c_index:
        raise ValueError("Full model must improve validation C-index over base.")
    target_c_index = base_c_index + retention_target * (full_c_index - base_c_index)
    feasible = [
        point for point in search_points if point.validation_c_index >= target_c_index
    ]
    if not feasible:
        raise ValueError("No adaptive cost penalty satisfies the retention target.")
    selected = min(
        feasible,
        key=lambda point: (
            point.mean_primary_cost,
            -point.validation_c_index,
            point.validation_ipcw_brier,
            -point.cost_penalty,
        ),
    )
    retained = (selected.validation_c_index - base_c_index) / (
        full_c_index - base_c_index
    )
    return SelectedAdaptiveOperatingPoint(
        point=selected,
        retention_target=retention_target,
        target_validation_c_index=target_c_index,
        base_validation_c_index=base_c_index,
        full_validation_c_index=full_c_index,
        retained_improvement_fraction=float(retained),
    )


def adaptive_prediction_metrics(
    simulation: AdaptivePolicySimulation,
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


def _action_name(action: int) -> str:
    if action == STOP_INDEX:
        return "STOP"
    if 0 <= action < STOP_INDEX:
        return CORE_ACQUIRABLE_FEATURES[action]
    raise ValueError(f"Invalid action index: {action}")


def adaptive_acquisition_summary(
    simulation: AdaptivePolicySimulation,
    frame: pd.DataFrame,
    costs: CostConfig,
    *,
    top_path_count: int = 20,
) -> dict[str, Any]:

    if len(frame) != len(simulation.final_masks):
        raise ValueError("Adaptive paths and cohort frame must have equal lengths.")
    acquired_count = simulation.acquired_count
    count_distribution = []
    for count in range(len(CORE_ACQUIRABLE_FEATURES) + 1):
        participants = int(np.sum(acquired_count == count))
        count_distribution.append(
            {
                "acquired_count": count,
                "participants": participants,
                "percent": float(100.0 * participants / len(frame)),
            }
        )

    action_by_step = []
    for step in range(simulation.action_indices.shape[1]):
        step_actions = simulation.action_indices[:, step]
        rows = []
        for action in range(STOP_INDEX + 1):
            count = int(np.sum(step_actions == action))
            if count:
                rows.append(
                    {
                        "action": _action_name(action),
                        "participants": count,
                        "percent_of_all_participants": float(100.0 * count / len(frame)),
                    }
                )
        if rows:
            action_by_step.append({"step": step + 1, "actions": rows})

    acquired_by_feature = []
    missing_attempts = []
    for index, feature in enumerate(CORE_ACQUIRABLE_FEATURES):
        acquired = (simulation.final_masks & int(BIT_VALUES[index])) != 0
        attempted = int(acquired.sum())
        source_missing = acquired & frame[feature].isna().to_numpy()
        missing = int(source_missing.sum())
        acquired_by_feature.append(
            {
                "feature": feature,
                "participants": attempted,
                "percent": float(100.0 * attempted / len(frame)),
            }
        )
        missing_attempts.append(
            {
                "feature": feature,
                "attempted": attempted,
                "source_value_available": attempted - missing,
                "source_value_missing": missing,
                "missing_percent_of_attempts": (
                    float(100.0 * missing / attempted) if attempted else 0.0
                ),
            }
        )

    path_counter = Counter(_path_tuple(row) for row in simulation.action_indices)
    top_paths = [
        {
            "path": [_action_name(action) for action in path],
            "participants": count,
            "percent": float(100.0 * count / len(frame)),
        }
        for path, count in path_counter.most_common(top_path_count)
    ]
    cost_summary = {}
    for scheme_name in costs.schemes:
        per_mask = _cost_by_mask(costs, scheme=scheme_name)
        values = per_mask[simulation.final_masks]
        cost_summary[scheme_name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p25": float(np.quantile(values, 0.25)),
            "p75": float(np.quantile(values, 0.75)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return {
        "mean_acquired_tests": float(np.mean(acquired_count)),
        "median_acquired_tests": float(np.median(acquired_count)),
        "p25_acquired_tests": float(np.quantile(acquired_count, 0.25)),
        "p75_acquired_tests": float(np.quantile(acquired_count, 0.75)),
        "stop_without_labs_percent": float(100.0 * np.mean(acquired_count == 0)),
        "acquire_all_percent": float(
            100.0 * np.mean(acquired_count == len(CORE_ACQUIRABLE_FEATURES))
        ),
        "unique_paths": len(path_counter),
        "unique_final_subsets": int(len(np.unique(simulation.final_masks))),
        "count_distribution": count_distribution,
        "action_by_step": action_by_step,
        "acquired_by_feature": acquired_by_feature,
        "top_paths": top_paths,
        "cost_by_scheme": cost_summary,
        "attempted_source_missingness": missing_attempts,
        "missing_result_rule": (
            "charge the selected action; use the train-fitted imputation value; "
            "retain the participant and expose source availability through its mask"
        ),
    }


def save_adaptive_value_artifact(
    value_models: FittedAdaptiveValueModels,
    selected: SelectedAdaptiveOperatingPoint,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": "adaptive_value_policy_v1",
            "policy_name": POLICY_NAME,
            "candidate_features": CORE_ACQUIRABLE_FEATURES,
            "state_feature_names": value_models.state_feature_names,
            "value_models": value_models.models,
            "training_seed": value_models.seed,
            "ridge_alpha": value_models.alpha,
            "selected_cost_penalty": selected.point.cost_penalty,
            "retention_target": selected.retention_target,
        },
        output_path,
    )
