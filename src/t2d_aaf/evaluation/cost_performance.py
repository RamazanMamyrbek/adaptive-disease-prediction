from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

from t2d_aaf.evaluation.survival_metrics import (
    harrell_c_index,
    kaplan_meier_curve,
    observed_event_risk_at_horizon,
)


@dataclass(frozen=True)
class HorizonOutcome:

    labels: NDArray[np.int64]
    weights: NDArray[np.float64]
    included: NDArray[np.bool_]


def _numeric_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not len(vector):
        raise ValueError(f"{name} must be a non-empty one-dimensional vector.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values.")
    return vector


def _event_vector(values: ArrayLike, name: str) -> NDArray[np.int64]:
    vector = _numeric_vector(values, name)
    if not np.isin(vector, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only 0 and 1.")
    return vector.astype(np.int64)


def _validate_lengths(**vectors: NDArray[np.generic]) -> None:
    lengths = {name: len(vector) for name, vector in vectors.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Vector lengths differ: {lengths}")


def fixed_horizon_ipcw_outcome(
    train_duration: ArrayLike,
    train_event: ArrayLike,
    evaluation_duration: ArrayLike,
    evaluation_event: ArrayLike,
    horizon: float,
    *,
    minimum_censoring_survival: float = 1e-8,
) -> HorizonOutcome:

    train_durations = _numeric_vector(train_duration, "train_duration")
    train_events = _event_vector(train_event, "train_event")
    durations = _numeric_vector(evaluation_duration, "evaluation_duration")
    events = _event_vector(evaluation_event, "evaluation_event")
    _validate_lengths(train_duration=train_durations, train_event=train_events)
    _validate_lengths(evaluation_duration=durations, evaluation_event=events)
    if horizon <= 0.0:
        raise ValueError("horizon must be positive.")
    if (train_durations <= 0.0).any() or (durations <= 0.0).any():
        raise ValueError("Durations must be strictly positive.")

    censoring_curve = kaplan_meier_curve(train_durations, 1 - train_events)
    censoring_at_horizon = censoring_curve.survival_at(horizon)
    labels = np.zeros(len(durations), dtype=np.int64)
    weights = np.zeros(len(durations), dtype=float)

    cases = (events == 1) & (durations <= horizon)
    labels[cases] = 1
    for index in np.flatnonzero(cases):
        censoring_probability = censoring_curve.survival_at(
            float(durations[index]), left_limit=True
        )
        if censoring_probability <= minimum_censoring_survival:
            raise ValueError("Training censoring survival is zero at a case time.")
        weights[index] = 1.0 / censoring_probability

    controls = durations > horizon
    if controls.any():
        if censoring_at_horizon <= minimum_censoring_survival:
            raise ValueError("Training censoring survival is zero at the horizon.")
        weights[controls] = 1.0 / censoring_at_horizon

    included = cases | controls
    if not cases.any() or not controls.any():
        raise ValueError("Horizon evaluation requires at least one case and one control.")
    return HorizonOutcome(labels=labels, weights=weights, included=included)


def fixed_horizon_discrimination(
    outcome: HorizonOutcome,
    predicted_risk: ArrayLike,
) -> dict[str, float | int]:

    risk = _numeric_vector(predicted_risk, "predicted_risk")
    _validate_lengths(labels=outcome.labels, weights=outcome.weights, risk=risk)
    if ((risk < 0.0) | (risk > 1.0)).any():
        raise ValueError("predicted_risk must lie in [0, 1].")
    included = outcome.included
    labels = outcome.labels[included]
    weights = outcome.weights[included]
    return {
        "ipcw_cumulative_dynamic_auc": float(
            roc_auc_score(labels, risk[included], sample_weight=weights)
        ),
        "ipcw_average_precision": float(
            average_precision_score(labels, risk[included], sample_weight=weights)
        ),
        "horizon_cases": int(labels.sum()),
        "horizon_controls": int(len(labels) - labels.sum()),
        "excluded_censored_before_or_at_horizon": int((~included).sum()),
    }


def ipcw_brier_contributions(
    train_duration: ArrayLike,
    train_event: ArrayLike,
    evaluation_duration: ArrayLike,
    evaluation_event: ArrayLike,
    predicted_risk: ArrayLike,
    horizon: float,
) -> NDArray[np.float64]:

    outcome = fixed_horizon_ipcw_outcome(
        train_duration,
        train_event,
        evaluation_duration,
        evaluation_event,
        horizon,
    )
    risk = _numeric_vector(predicted_risk, "predicted_risk")
    _validate_lengths(labels=outcome.labels, risk=risk)
    if ((risk < 0.0) | (risk > 1.0)).any():
        raise ValueError("predicted_risk must lie in [0, 1].")
    residual = outcome.labels.astype(float) - risk
    return outcome.weights * residual**2


def calibration_bins(
    duration: ArrayLike,
    event: ArrayLike,
    predicted_risk: ArrayLike,
    horizon: float,
    *,
    bin_count: int = 10,
) -> tuple[dict[str, float | int], ...]:

    durations = _numeric_vector(duration, "duration")
    events = _event_vector(event, "event")
    risk = _numeric_vector(predicted_risk, "predicted_risk")
    _validate_lengths(duration=durations, event=events, risk=risk)
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2.")
    if ((risk < 0.0) | (risk > 1.0)).any():
        raise ValueError("predicted_risk must lie in [0, 1].")

    ordered_rows = np.argsort(risk, kind="mergesort")
    bins: list[dict[str, float | int]] = []
    for bin_index, rows in enumerate(np.array_split(ordered_rows, bin_count), start=1):
        if not len(rows):
            continue
        observed = observed_event_risk_at_horizon(
            durations[rows], events[rows], horizon
        )
        predicted = float(np.mean(risk[rows]))
        bins.append(
            {
                "bin": bin_index,
                "participants": int(len(rows)),
                "events_total": int(events[rows].sum()),
                "mean_predicted_risk": predicted,
                "km_observed_risk": observed,
                "absolute_gap": abs(predicted - observed),
                "minimum_predicted_risk": float(np.min(risk[rows])),
                "maximum_predicted_risk": float(np.max(risk[rows])),
            }
        )
    return tuple(bins)


def prediction_evaluation(
    train_duration: ArrayLike,
    train_event: ArrayLike,
    evaluation_duration: ArrayLike,
    evaluation_event: ArrayLike,
    predicted_risk: ArrayLike,
    horizon: float,
    *,
    calibration_bin_count: int = 10,
) -> dict[str, object]:

    train_durations = _numeric_vector(train_duration, "train_duration")
    train_events = _event_vector(train_event, "train_event")
    durations = _numeric_vector(evaluation_duration, "evaluation_duration")
    events = _event_vector(evaluation_event, "evaluation_event")
    risk = _numeric_vector(predicted_risk, "predicted_risk")
    _validate_lengths(duration=durations, event=events, risk=risk)
    horizon_outcome = fixed_horizon_ipcw_outcome(
        train_durations,
        train_events,
        durations,
        events,
        horizon,
    )
    discrimination = fixed_horizon_discrimination(horizon_outcome, risk)
    bins = calibration_bins(
        durations,
        events,
        risk,
        horizon,
        bin_count=calibration_bin_count,
    )
    weighted_absolute_gap = sum(
        int(row["participants"]) * float(row["absolute_gap"]) for row in bins
    ) / len(risk)
    observed = observed_event_risk_at_horizon(durations, events, horizon)
    predicted = float(np.mean(risk))
    return {
        "participants": int(len(risk)),
        "events_total": int(events.sum()),
        "harrell_c_index": harrell_c_index(durations, events, risk),
        "ipcw_brier_at_horizon": float(
            np.mean(
                ipcw_brier_contributions(
                    train_durations,
                    train_events,
                    durations,
                    events,
                    risk,
                    horizon,
                )
            )
        ),
        **discrimination,
        "mean_predicted_risk_at_horizon": predicted,
        "km_observed_risk_at_horizon": observed,
        "calibration_in_the_large_difference": predicted - observed,
        "calibration_mean_absolute_bin_gap": float(weighted_absolute_gap),
        "calibration_maximum_absolute_bin_gap": float(
            max(float(row["absolute_gap"]) for row in bins)
        ),
        "calibration_bins": list(bins),
    }


def stratified_bootstrap_indices(
    event: ArrayLike,
    *,
    replicates: int,
    seed: int,
) -> tuple[NDArray[np.int64], ...]:

    events = _event_vector(event, "event")
    if replicates < 1:
        raise ValueError("replicates must be positive.")
    strata = tuple(np.flatnonzero(events == value) for value in (0, 1))
    if any(not len(rows) for rows in strata):
        raise ValueError("Both event strata must be present for stratified bootstrap.")
    rng = np.random.default_rng(seed)
    samples: list[NDArray[np.int64]] = []
    for _ in range(replicates):
        sampled = np.concatenate(
            [rng.choice(rows, size=len(rows), replace=True) for rows in strata]
        ).astype(np.int64)
        rng.shuffle(sampled)
        samples.append(sampled)
    return tuple(samples)


def _interval(values: Sequence[float], point: float) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    valid = array[np.isfinite(array)]
    if not len(valid):
        raise ValueError("No finite bootstrap estimates are available.")
    return {
        "point_estimate": float(point),
        "ci95_lower": float(np.quantile(valid, 0.025)),
        "ci95_upper": float(np.quantile(valid, 0.975)),
        "valid_replicates": int(len(valid)),
    }


def paired_bootstrap_evaluation(
    train_duration: ArrayLike,
    train_event: ArrayLike,
    evaluation_duration: ArrayLike,
    evaluation_event: ArrayLike,
    method_risks: Mapping[str, ArrayLike],
    method_costs: Mapping[str, ArrayLike],
    horizon: float,
    *,
    contrasts: Sequence[tuple[str, str]] = (),
    replicates: int = 200,
    seed: int = 20260827,
) -> dict[str, object]:

    train_durations = _numeric_vector(train_duration, "train_duration")
    train_events = _event_vector(train_event, "train_event")
    durations = _numeric_vector(evaluation_duration, "evaluation_duration")
    events = _event_vector(evaluation_event, "evaluation_event")
    risks = {
        name: _numeric_vector(values, f"method_risks[{name}]")
        for name, values in method_risks.items()
    }
    costs = {
        name: _numeric_vector(values, f"method_costs[{name}]")
        for name, values in method_costs.items()
    }
    if set(risks) != set(costs):
        raise ValueError("Risk and cost method names must match.")
    for name in risks:
        _validate_lengths(duration=durations, risk=risks[name], cost=costs[name])
    for left, right in contrasts:
        if left not in risks or right not in risks:
            raise KeyError(f"Unknown bootstrap contrast: {(left, right)}")

    horizon_outcome = fixed_horizon_ipcw_outcome(
        train_durations,
        train_events,
        durations,
        events,
        horizon,
    )
    brier = {
        name: ipcw_brier_contributions(
            train_durations,
            train_events,
            durations,
            events,
            risk,
            horizon,
        )
        for name, risk in risks.items()
    }
    metric_names = (
        "harrell_c_index",
        "ipcw_brier_at_horizon",
        "ipcw_cumulative_dynamic_auc",
        "ipcw_average_precision",
        "mean_cost",
    )
    replicate_values = {
        name: {metric: [] for metric in metric_names} for name in risks
    }
    samples = stratified_bootstrap_indices(events, replicates=replicates, seed=seed)
    for rows in samples:
        included = horizon_outcome.included[rows]
        horizon_labels = horizon_outcome.labels[rows][included]
        horizon_weights = horizon_outcome.weights[rows][included]
        for name, risk in risks.items():
            sampled_risk = risk[rows]
            values = replicate_values[name]
            values["harrell_c_index"].append(
                harrell_c_index(durations[rows], events[rows], sampled_risk)
            )
            values["ipcw_brier_at_horizon"].append(float(np.mean(brier[name][rows])))
            values["ipcw_cumulative_dynamic_auc"].append(
                float(
                    roc_auc_score(
                        horizon_labels,
                        sampled_risk[included],
                        sample_weight=horizon_weights,
                    )
                )
            )
            values["ipcw_average_precision"].append(
                float(
                    average_precision_score(
                        horizon_labels,
                        sampled_risk[included],
                        sample_weight=horizon_weights,
                    )
                )
            )
            values["mean_cost"].append(float(np.mean(costs[name][rows])))

    point_metrics: dict[str, dict[str, float]] = {}
    method_intervals: dict[str, dict[str, dict[str, float | int]]] = {}
    for name, risk in risks.items():
        discrimination = fixed_horizon_discrimination(horizon_outcome, risk)
        points = {
            "harrell_c_index": harrell_c_index(durations, events, risk),
            "ipcw_brier_at_horizon": float(np.mean(brier[name])),
            "ipcw_cumulative_dynamic_auc": float(
                discrimination["ipcw_cumulative_dynamic_auc"]
            ),
            "ipcw_average_precision": float(discrimination["ipcw_average_precision"]),
            "mean_cost": float(np.mean(costs[name])),
        }
        point_metrics[name] = points
        method_intervals[name] = {
            metric: _interval(replicate_values[name][metric], points[metric])
            for metric in metric_names
        }

    contrast_intervals: dict[str, dict[str, dict[str, float | int]]] = {}
    for left, right in contrasts:
        key = f"{left}_minus_{right}"
        contrast_intervals[key] = {}
        for metric in metric_names:
            differences = np.asarray(replicate_values[left][metric]) - np.asarray(
                replicate_values[right][metric]
            )
            point = point_metrics[left][metric] - point_metrics[right][metric]
            contrast_intervals[key][metric] = _interval(differences, point)

    return {
        "method": "participant-level event-stratified paired percentile bootstrap",
        "seed": seed,
        "replicates_requested": replicates,
        "stratification": "event status within the unchanged test split",
        "training_censoring_distribution_refit_per_replicate": False,
        "methods": method_intervals,
        "contrasts": contrast_intervals,
    }
