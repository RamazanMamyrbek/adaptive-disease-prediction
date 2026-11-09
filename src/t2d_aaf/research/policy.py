from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from t2d_aaf.acquisition.adaptive import (
    BIT_VALUES, PreparedAdaptiveData, _sample_training_masks, build_state_matrix,
)


PANEL_MASKS = np.array([mask for mask in range(1, 64) if mask.bit_count() > 1], dtype=np.int64)


@dataclass(frozen=True)
class MarginalCosts:
    tests: tuple[float, ...] = (1.0,) * 6
    draw: float = 0.0

    def __post_init__(self):
        if len(self.tests) != 6 or not np.isfinite(self.tests).all() or min(self.tests) <= 0:
            raise ValueError('Six finite, positive test costs are required.')
        if not np.isfinite(self.draw) or self.draw < 0:
            raise ValueError('Draw cost must be finite and nonnegative.')

    def total(self, masks):
        masks = np.asarray(masks, dtype=np.int64)
        if np.any((masks < 0) | (masks > 63)):
            raise ValueError('Invalid acquisition mask.')
        return ((masks[..., None] & BIT_VALUES) != 0) @ np.asarray(self.tests) + (masks != 0) * self.draw

    def delta(self, masks, additions):
        return self.total(np.asarray(masks) | np.asarray(additions)) - self.total(masks)


@dataclass(frozen=True)
class UtilityModels:
    criterion: str
    steps: tuple[Ridge, ...]
    panels: Ridge
    training_summary: dict


@dataclass(frozen=True)
class ScoreTable:
    steps: np.ndarray
    panels: np.ndarray


@dataclass(frozen=True)
class PolicyRun:
    masks: np.ndarray
    initial_masks: np.ndarray


def utility_targets(current, following, criterion, labels=None, weights=None):
    if criterion == 'revision':
        return np.abs(following - current)
    if criterion != 'brier' or labels is None or weights is None:
        raise ValueError('Brier reduction requires censoring-adjusted outcomes.')
    return weights * ((labels - current) ** 2 - (labels - following) ** 2)


def fit_utilities(prepared: PreparedAdaptiveData, criterion: str, *, labels=None, weights=None,
                  seed=20260827, alpha=1.0):
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError('Ridge penalty must be finite and nonnegative.')
    rng = np.random.default_rng(seed)
    rows = np.arange(prepared.participants)
    steps, summaries = [], []
    for candidate in range(6):
        masks = _sample_training_masks(rng, len(rows), candidate)
        target = utility_targets(prepared.subset_risks[rows, masks],
                                 prepared.subset_risks[rows, masks | int(BIT_VALUES[candidate])],
                                 criterion, labels, weights)
        estimator = Ridge(alpha=alpha).fit(build_state_matrix(prepared, masks), target)
        steps.append(estimator)
        summaries.append({'candidate_index': candidate, 'target_mean': float(target.mean()),
                          'target_min': float(target.min()), 'target_max': float(target.max()),
                          'negative_target_fraction': float(np.mean(target < 0))})
    current = prepared.subset_risks[:, [0]]
    following = prepared.subset_risks[:, PANEL_MASKS]
    target = utility_targets(current, following, criterion,
                             None if labels is None else labels[:, None],
                             None if weights is None else weights[:, None])
    panels = Ridge(alpha=alpha).fit(build_state_matrix(prepared, np.zeros(len(rows), dtype=int)), target)
    return UtilityModels(criterion, tuple(steps), panels,
                         {'steps': summaries, 'panel_masks': PANEL_MASKS.tolist(),
                          'panel_negative_target_fraction': float(np.mean(target < 0)),
                          'panel_target': 'direct base-to-panel gain; no future feature values in inputs',
                          'seed': seed, 'ridge_alpha': alpha})


def predict_gain(values, criterion):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite predicted utility.')
    return np.clip(values, 0.0, 1.0) if criterion == 'revision' else values


def score_table(prepared, model):
    count = prepared.participants
    scores = np.empty((count, 64, 6))
    coefficients = np.stack([item.coef_ for item in model.steps])
    intercepts = np.array([item.intercept_ for item in model.steps])
    for mask in range(64):
        state = build_state_matrix(prepared, np.full(count, mask, dtype=int))
        scores[:, mask] = predict_gain(state @ coefficients.T + intercepts, model.criterion)
    base = build_state_matrix(prepared, np.zeros(count, dtype=int))
    return ScoreTable(scores, predict_gain(model.panels.predict(base), model.criterion))


def simulate(scores: ScoreTable, costs: MarginalCosts, penalty: float, *, panels=False, available=63):
    if np.isnan(penalty) or penalty < 0 or not 0 < available <= 63:
        raise ValueError('Invalid penalty or available mask.')
    count = len(scores.steps)
    masks, initial = np.zeros(count, dtype=np.int64), np.zeros(count, dtype=np.int64)
    if np.isinf(penalty):
        return PolicyRun(masks, initial)
    active = np.ones(count, dtype=bool)
    allowed = (BIT_VALUES & available) != 0
    panel_allowed = (PANEL_MASKS & available) == PANEL_MASKS
    for step in range(6):
        rows = np.flatnonzero(active)
        if not len(rows):
            break
        state = masks[rows]
        utilities = scores.steps[rows, state] - penalty * costs.delta(state[:, None], BIT_VALUES)
        utilities[(state[:, None] & BIT_VALUES) != 0] = -np.inf
        utilities[:, ~allowed] = -np.inf
        choices = np.argmax(utilities, axis=1)
        best = utilities[np.arange(len(rows)), choices]
        additions = BIT_VALUES[choices].copy()
        if step == 0 and panels and panel_allowed.any():
            joint = scores.panels[rows] - penalty * costs.total(PANEL_MASKS)
            joint[:, ~panel_allowed] = -np.inf
            choices = np.argmax(joint, axis=1)
            panel_best = joint[np.arange(len(rows)), choices]
            replace = panel_best > best
            additions[replace] = PANEL_MASKS[choices[replace]]
            best[replace] = panel_best[replace]
        acquire = best > 0
        masks[rows[acquire]] |= additions[acquire]
        active[rows[~acquire]] = False
        if step == 0:
            initial[:] = masks
    return PolicyRun(masks, initial)


def penalty_grid(prepared, model, costs, *, quantiles=31, panels=False, seed=20260827):
    if quantiles < 3:
        raise ValueError('At least three quantiles are required.')
    rng = np.random.default_rng(seed)
    ratios = []
    for candidate, estimator in enumerate(model.steps):
        for masks in (_sample_training_masks(rng, prepared.participants, candidate),
                      np.zeros(prepared.participants, dtype=int)):
            gains = predict_gain(estimator.predict(build_state_matrix(prepared, masks)), model.criterion)
            ratios.append(gains / costs.delta(masks, int(BIT_VALUES[candidate])))
    if panels:
        state = build_state_matrix(prepared, np.zeros(prepared.participants, dtype=int))
        ratios.append((predict_gain(model.panels.predict(state), model.criterion) / costs.total(PANEL_MASKS)).ravel())
    positive = np.concatenate(ratios)
    positive = positive[positive > 0]
    if not len(positive):
        return np.array([0.0, np.inf])
    return np.unique(np.r_[0.0, np.quantile(positive, np.linspace(0, 1, quantiles)), np.inf])
