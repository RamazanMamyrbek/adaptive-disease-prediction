from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
import random
import warnings

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.compose import ColumnTransformer

from t2d_aaf.data.cohort import BASE_FEATURES, CORE_ACQUIRABLE_FEATURES
from t2d_aaf.evaluation.survival_metrics import (
    harrell_c_index,
    ipcw_brier_score,
    observed_event_risk_at_horizon,
)
from t2d_aaf.survival.baseline import (
    EVALUATION_HORIZON_YEARS,
    PREDICTOR_FEATURES,
    TIES_METHOD,
    CohortSplits,
    baseline_hazard_at,
    breslow_baseline_cumulative_hazard,
    build_preprocessor,
    transform_predictors,
)


STATIC_RANDOM_SEED = 20260827
DEFAULT_RANDOM_REPEATS = 1000
EVALUATION_SPLITS: tuple[str, ...] = ("validation", "test")
OPTIMIZER = "bfgs"


@dataclass(frozen=True)
class FittedStaticSubset:
    acquired_features: tuple[str, ...]
    selected_feature_names: tuple[str, ...]
    coefficients: NDArray[np.float64]
    baseline_event_times: NDArray[np.float64]
    baseline_cumulative_hazard: NDArray[np.float64]
    risk_center: float
    fit_warnings: tuple[str, ...]


@dataclass(frozen=True)
class StaticSubsetRun:
    preprocessor: ColumnTransformer
    transformed_feature_names: tuple[str, ...]
    fitted_models: Mapping[tuple[str, ...], FittedStaticSubset]
    evaluations: Mapping[tuple[str, ...], Mapping[str, Mapping[str, float | int]]]


def canonical_subset(
    features: Sequence[str],
    candidates: Sequence[str] = CORE_ACQUIRABLE_FEATURES,
) -> tuple[str, ...]:

    requested = tuple(features)
    if len(requested) != len(set(requested)):
        raise ValueError("A static feature subset cannot contain duplicates.")
    unknown = sorted(set(requested) - set(candidates))
    if unknown:
        raise KeyError(f"Unknown static acquisition candidates: {unknown}")
    return tuple(feature for feature in candidates if feature in requested)


def subset_id(features: Sequence[str]) -> str:
    canonical = canonical_subset(features)
    return "base_only" if not canonical else "+".join(canonical)


def enumerate_candidate_subsets(
    candidates: Sequence[str] = CORE_ACQUIRABLE_FEATURES,
) -> tuple[tuple[str, ...], ...]:

    ordered = tuple(candidates)
    if len(ordered) != len(set(ordered)):
        raise ValueError("Candidate names must be unique.")
    return tuple(
        subset
        for size in range(len(ordered) + 1)
        for subset in combinations(ordered, size)
    )


def singleton_validation_order(
    validation_scores: Mapping[tuple[str, ...], float],
    candidates: Sequence[str] = CORE_ACQUIRABLE_FEATURES,
) -> tuple[str, ...]:

    candidate_order = {feature: index for index, feature in enumerate(candidates)}
    for feature in candidates:
        score = validation_scores.get(canonical_subset((feature,), candidates))
        if score is None or not np.isfinite(score):
            raise ValueError(f"Missing finite singleton validation score for {feature}.")
    return tuple(
        sorted(
            candidates,
            key=lambda feature: (
                -validation_scores[canonical_subset((feature,), candidates)],
                candidate_order[feature],
            ),
        )
    )


def greedy_validation_order(
    validation_scores: Mapping[tuple[str, ...], float],
    candidates: Sequence[str] = CORE_ACQUIRABLE_FEATURES,
) -> tuple[str, ...]:

    selected: list[str] = []
    remaining = list(candidates)
    candidate_order = {feature: index for index, feature in enumerate(candidates)}
    while remaining:
        scored: list[tuple[float, int, str]] = []
        for feature in remaining:
            subset = canonical_subset((*selected, feature), candidates)
            score = validation_scores.get(subset)
            if score is None or not np.isfinite(score):
                raise ValueError(f"Missing finite validation score for subset {subset}.")
            scored.append((-score, candidate_order[feature], feature))
        chosen = min(scored)[2]
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)


def derive_static_orders(
    evaluations: Mapping[tuple[str, ...], Mapping[str, Mapping[str, float | int]]],
    candidates: Sequence[str] = CORE_ACQUIRABLE_FEATURES,
) -> dict[str, tuple[str, ...]]:

    validation_scores = {
        subset: float(split_metrics["validation"]["harrell_c_index"])
        for subset, split_metrics in evaluations.items()
    }
    return {
        "static_top_k": singleton_validation_order(validation_scores, candidates),
        "importance_greedy": greedy_validation_order(validation_scores, candidates),
    }


def prefix_subsets(
    order: Sequence[str],
    candidates: Sequence[str] = CORE_ACQUIRABLE_FEATURES,
) -> tuple[tuple[str, ...], ...]:
    if set(order) != set(candidates) or len(order) != len(candidates):
        raise ValueError("A static order must contain every candidate exactly once.")
    return tuple(canonical_subset(order[:size], candidates) for size in range(len(order) + 1))


def generate_random_orders(
    *,
    repeats: int = DEFAULT_RANDOM_REPEATS,
    seed: int = STATIC_RANDOM_SEED,
    candidates: Sequence[str] = CORE_ACQUIRABLE_FEATURES,
) -> tuple[tuple[str, ...], ...]:
    if repeats <= 0:
        raise ValueError("Random baseline repeats must be positive.")
    ordered = tuple(candidates)
    if len(ordered) != len(set(ordered)):
        raise ValueError("Candidate names must be unique.")
    rng = random.Random(seed)
    paths: list[tuple[str, ...]] = []
    for _ in range(repeats):
        path = list(ordered)
        rng.shuffle(path)
        paths.append(tuple(path))
    return tuple(paths)


def _selected_matrix_indices(
    transformed_feature_names: Sequence[str],
    acquired_features: Sequence[str],
) -> tuple[int, ...]:
    selected = set(BASE_FEATURES) | set(canonical_subset(acquired_features))
    indices = tuple(
        index for index, feature in enumerate(transformed_feature_names) if feature in selected
    )
    observed = {transformed_feature_names[index] for index in indices}
    if observed != selected:
        raise ValueError(f"Transformed predictor mismatch: expected={selected}, observed={observed}")
    return indices


def fit_static_subset(
    train_matrix: NDArray[np.float64],
    train_frame: pd.DataFrame,
    transformed_feature_names: Sequence[str],
    acquired_features: Sequence[str],
) -> FittedStaticSubset:

    from statsmodels.duration.hazard_regression import PHReg

    acquired = canonical_subset(acquired_features)
    indices = _selected_matrix_indices(transformed_feature_names, acquired)
    matrix = train_matrix[:, indices]
    duration = train_frame["duration_years"].to_numpy(dtype=float)
    event = train_frame["event"].to_numpy(dtype=np.int64)
    model = PHReg(duration, matrix, status=event, ties=TIES_METHOD, missing="raise")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(method=OPTIMIZER, disp=False, maxiter=500)
    coefficients = np.asarray(result.params, dtype=float)
    if not np.isfinite(coefficients).all():
        raise ValueError(f"Cox fit returned non-finite coefficients for {acquired}.")
    linear_predictor = matrix @ coefficients
    event_times, cumulative_hazard, risk_center = breslow_baseline_cumulative_hazard(
        duration,
        event,
        linear_predictor,
    )
    return FittedStaticSubset(
        acquired_features=acquired,
        selected_feature_names=tuple(transformed_feature_names[index] for index in indices),
        coefficients=coefficients,
        baseline_event_times=event_times,
        baseline_cumulative_hazard=cumulative_hazard,
        risk_center=risk_center,
        fit_warnings=tuple(str(item.message) for item in caught),
    )


def evaluate_static_subset(
    fitted: FittedStaticSubset,
    matrices: Mapping[str, NDArray[np.float64]],
    transformed_feature_names: Sequence[str],
    splits: CohortSplits,
    *,
    horizon: float = EVALUATION_HORIZON_YEARS,
) -> dict[str, dict[str, float | int]]:

    indices = _selected_matrix_indices(transformed_feature_names, fitted.acquired_features)
    train_duration = splits.train["duration_years"].to_numpy(dtype=float)
    train_event = splits.train["event"].to_numpy(dtype=np.int64)
    baseline_hazard = baseline_hazard_at(
        fitted.baseline_event_times,
        fitted.baseline_cumulative_hazard,
        horizon,
    )
    output: dict[str, dict[str, float | int]] = {}
    for split_name in EVALUATION_SPLITS:
        frame = splits.as_dict()[split_name]
        matrix = matrices[split_name][:, indices]
        linear_predictor = matrix @ fitted.coefficients
        relative_hazard = np.exp(linear_predictor - fitted.risk_center)
        predicted_survival = np.exp(-baseline_hazard * relative_hazard)
        duration = frame["duration_years"].to_numpy(dtype=float)
        event = frame["event"].to_numpy(dtype=np.int64)
        observed_risk = observed_event_risk_at_horizon(duration, event, horizon)
        predicted_risk = float(np.mean(1.0 - predicted_survival))
        output[split_name] = {
            "participants": int(len(frame)),
            "events_total": int(event.sum()),
            "harrell_c_index": harrell_c_index(duration, event, linear_predictor),
            "ipcw_brier_at_horizon": ipcw_brier_score(
                train_duration,
                train_event,
                duration,
                event,
                predicted_survival,
                horizon,
            ),
            "mean_predicted_risk_at_horizon": predicted_risk,
            "km_observed_risk_at_horizon": observed_risk,
            "calibration_in_the_large_difference": predicted_risk - observed_risk,
        }
    return output


def fit_all_static_subsets(
    splits: CohortSplits,
    *,
    progress: Callable[[int, int, tuple[str, ...]], None] | None = None,
) -> StaticSubsetRun:

    preprocessor = build_preprocessor()
    train_matrix = np.asarray(
        preprocessor.fit_transform(splits.train.loc[:, PREDICTOR_FEATURES]), dtype=float
    )
    if not np.isfinite(train_matrix).all():
        raise ValueError("Training preprocessing produced invalid predictor values.")
    matrices = {
        "train": train_matrix,
        "validation": transform_predictors(preprocessor, splits.validation),
        "test": transform_predictors(preprocessor, splits.test),
    }
    transformed_names = tuple(str(name) for name in preprocessor.get_feature_names_out())
    subsets = enumerate_candidate_subsets()
    fitted_models: dict[tuple[str, ...], FittedStaticSubset] = {}
    evaluations: dict[tuple[str, ...], Mapping[str, Mapping[str, float | int]]] = {}
    for index, subset in enumerate(subsets, start=1):
        if progress is not None:
            progress(index, len(subsets), subset)
        fitted = fit_static_subset(train_matrix, splits.train, transformed_names, subset)
        fitted_models[subset] = fitted
        evaluations[subset] = evaluate_static_subset(
            fitted,
            matrices,
            transformed_names,
            splits,
        )
    return StaticSubsetRun(
        preprocessor=preprocessor,
        transformed_feature_names=transformed_names,
        fitted_models=fitted_models,
        evaluations=evaluations,
    )
