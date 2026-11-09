from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _numeric_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values.")
    return vector


def _event_vector(values: ArrayLike, name: str = "event") -> NDArray[np.int64]:
    vector = _numeric_vector(values, name)
    if not np.isin(vector, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only 0 and 1.")
    return vector.astype(np.int64)


def _validate_equal_lengths(**vectors: NDArray[np.generic]) -> None:
    lengths = {name: len(vector) for name, vector in vectors.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Vector lengths differ: {lengths}")


@dataclass(frozen=True)
class KaplanMeierCurve:

    times: NDArray[np.float64]
    survival_after_time: NDArray[np.float64]

    def survival_at(self, time: float, *, left_limit: bool = False) -> float:

        side = "left" if left_limit else "right"
        index = int(np.searchsorted(self.times, time, side=side)) - 1
        if index < 0:
            return 1.0
        return float(self.survival_after_time[index])


def kaplan_meier_curve(duration: ArrayLike, event: ArrayLike) -> KaplanMeierCurve:

    durations = _numeric_vector(duration, "duration")
    events = _event_vector(event)
    _validate_equal_lengths(duration=durations, event=events)
    if (durations <= 0).any():
        raise ValueError("duration must be strictly positive.")

    order = np.argsort(durations, kind="mergesort")
    sorted_duration = durations[order]
    sorted_event = events[order]
    unique_times, starts = np.unique(sorted_duration, return_index=True)
    ends = np.r_[starts[1:], len(sorted_duration)]

    at_risk = len(sorted_duration)
    survival = 1.0
    values: list[float] = []
    for start, end in zip(starts, ends, strict=True):
        event_count = int(sorted_event[start:end].sum())
        if event_count:
            survival *= 1.0 - event_count / at_risk
        values.append(survival)
        at_risk -= int(end - start)

    return KaplanMeierCurve(
        times=unique_times.astype(float),
        survival_after_time=np.asarray(values, dtype=float),
    )


class _FenwickTree:
    def __init__(self, size: int) -> None:
        self._tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, zero_based_index: int) -> None:
        index = zero_based_index + 1
        while index < len(self._tree):
            self._tree[index] += 1
            index += index & -index

    def prefix_count(self, item_count: int) -> int:

        total = 0
        index = item_count
        while index > 0:
            total += int(self._tree[index])
            index -= index & -index
        return total


def harrell_c_index(duration: ArrayLike, event: ArrayLike, risk_score: ArrayLike) -> float:

    durations = _numeric_vector(duration, "duration")
    events = _event_vector(event)
    scores = _numeric_vector(risk_score, "risk_score")
    _validate_equal_lengths(duration=durations, event=events, risk_score=scores)
    if (durations <= 0).any():
        raise ValueError("duration must be strictly positive.")

    unique_scores = np.unique(scores)
    ranks = np.searchsorted(unique_scores, scores)
    order = np.argsort(-durations, kind="mergesort")
    tree = _FenwickTree(len(unique_scores))

    comparable = 0
    concordant = 0.0
    inserted = 0
    start = 0
    while start < len(order):
        end = start + 1
        current_time = durations[order[start]]
        while end < len(order) and durations[order[end]] == current_time:
            end += 1

        for row_index in order[start:end]:
            if events[row_index] != 1:
                continue
            rank = int(ranks[row_index])
            lower = tree.prefix_count(rank)
            lower_or_equal = tree.prefix_count(rank + 1)
            equal = lower_or_equal - lower
            concordant += lower + 0.5 * equal
            comparable += inserted

        for row_index in order[start:end]:
            tree.add(int(ranks[row_index]))
        inserted += end - start
        start = end

    if comparable == 0:
        raise ValueError("No comparable survival pairs are available.")
    return concordant / comparable


def observed_event_risk_at_horizon(duration: ArrayLike, event: ArrayLike, horizon: float) -> float:

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    return 1.0 - kaplan_meier_curve(duration, event).survival_at(horizon)


def ipcw_brier_score(
    train_duration: ArrayLike,
    train_event: ArrayLike,
    evaluation_duration: ArrayLike,
    evaluation_event: ArrayLike,
    predicted_survival: ArrayLike,
    horizon: float,
    *,
    minimum_censoring_survival: float = 1e-8,
) -> float:

    train_durations = _numeric_vector(train_duration, "train_duration")
    train_events = _event_vector(train_event, "train_event")
    durations = _numeric_vector(evaluation_duration, "evaluation_duration")
    events = _event_vector(evaluation_event, "evaluation_event")
    survival = _numeric_vector(predicted_survival, "predicted_survival")
    _validate_equal_lengths(train_duration=train_durations, train_event=train_events)
    _validate_equal_lengths(
        evaluation_duration=durations,
        evaluation_event=events,
        predicted_survival=survival,
    )
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if ((survival < 0) | (survival > 1)).any():
        raise ValueError("predicted_survival must lie in [0, 1].")

    censoring_curve = kaplan_meier_curve(train_durations, 1 - train_events)
    censoring_at_horizon = censoring_curve.survival_at(horizon)
    contributions = np.zeros(len(durations), dtype=float)

    cases = (events == 1) & (durations <= horizon)
    for index in np.flatnonzero(cases):
        censoring_probability = censoring_curve.survival_at(
            float(durations[index]), left_limit=True
        )
        if censoring_probability <= minimum_censoring_survival:
            raise ValueError("Training censoring survival is zero at an event time.")
        contributions[index] = survival[index] ** 2 / censoring_probability

    controls = durations > horizon
    if controls.any():
        if censoring_at_horizon <= minimum_censoring_survival:
            raise ValueError("Training censoring survival is zero at the evaluation horizon.")
        contributions[controls] = (1.0 - survival[controls]) ** 2 / censoring_at_horizon

    return float(contributions.mean())
