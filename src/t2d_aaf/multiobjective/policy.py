from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from t2d_aaf.acquisition.adaptive import BIT_VALUES, build_state_matrix, _sample_training_masks
from t2d_aaf.clinical.thresholds import LAB_BITS
from t2d_aaf.research.policy import predict_gain


@dataclass
class ProbabilityModel:
    estimator: object | None
    constant: float = 0.

    def predict(self, state):
        if self.estimator is None:
            return np.full(len(state), self.constant)
        return self.estimator.predict_proba(state)[:, 1]


@dataclass
class LabModels:
    abnormal: tuple[ProbabilityModel, ...]
    observed: tuple[ProbabilityModel, ...]
    summary: list[dict]


def fit_binary(state, labels):
    if not len(labels):
        raise ValueError('No observed labels for probability training.')
    if len(np.unique(labels)) == 1:
        return ProbabilityModel(None, float(labels[0]))
    model = LogisticRegression(C=1., max_iter=600, solver='lbfgs').fit(state, labels)
    if model.n_iter_.max() >= 600:
        raise RuntimeError('Lab probability model did not converge.')
    return ProbabilityModel(model)


def fit_labs(prepared, labels, seed=20261110):
    rng = np.random.default_rng(seed)
    abnormal, observed, summary = [], [], []
    for target, bit in enumerate(LAB_BITS):
        candidate = int(bit).bit_length() - 1
        masks = _sample_training_masks(rng, prepared.participants, candidate)
        state = np.concatenate((build_state_matrix(prepared, masks),
                                build_state_matrix(prepared, np.zeros(len(masks), dtype=int))))
        outcome = np.tile(labels[:, target], 2)
        known = np.isfinite(outcome)
        abnormal.append(fit_binary(state[known], outcome[known]))
        observed.append(fit_binary(state, known.astype(int)))
        summary.append({'target_bit': int(bit), 'observed_participants': int(known.sum() // 2),
                        'positive_participants': int(np.nansum(outcome) // 2),
                        'training_states_per_participant': 2, 'own_test_hidden': True})
    return LabModels(tuple(abnormal), tuple(observed), summary)


def lab_probabilities(prepared, masks, models):
    masks = np.asarray(masks, dtype=int)
    conditional, availability = [], []
    for bit, abnormal, observed in zip(LAB_BITS, models.abnormal, models.observed):
        state = build_state_matrix(prepared, masks & ~bit)
        conditional.append(abnormal.predict(state))
        availability.append(observed.predict(state))
    return np.column_stack(conditional), np.column_stack(availability)


def combined_table(prepared, survival, labs, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive training-only survival scale required.')
    n = prepared.participants
    revision, discovery = np.zeros((n, 64, 6), dtype=np.float32), np.zeros((n, 64, 6), dtype=np.float32)
    coef = np.stack([model.coef_ for model in survival.steps])
    intercept = np.array([model.intercept_ for model in survival.steps])
    for mask in range(64):
        masks = np.full(n, mask, dtype=int)
        state = build_state_matrix(prepared, masks)
        revision[:, mask] = predict_gain(state @ coef.T + intercept, survival.criterion) / scale
        for target, bit in enumerate(LAB_BITS):
            if not mask & bit:
                discovery[:, mask, int(bit).bit_length() - 1] = (
                    labs.abnormal[target].predict(state) * labs.observed[target].predict(state))
    return revision, discovery


def budget_policy(revision, discovery, weight, budget, costs, *, available=63, initial=None):
    if not np.isfinite(weight) or weight < 0 or not np.isfinite(budget) or budget < 0:
        raise ValueError('Invalid weight or budget.')
    n = len(revision)
    masks = np.zeros(n, dtype=int) if initial is None else np.asarray(initial, dtype=int).copy()
    if np.any(masks & ~available) or np.any(costs.total(masks) > budget + 1e-9):
        raise ValueError('Initial state exceeds available tests or budget.')
    rows = np.arange(n)
    for _ in range(6):
        delta = costs.delta(masks[:, None], BIT_VALUES)
        score = (revision[rows, masks] + weight * discovery[rows, masks]) / np.maximum(delta, 1e-12)
        allowed = ((masks[:, None] & BIT_VALUES) == 0) & ((available & BIT_VALUES) != 0)
        allowed &= costs.total(masks[:, None] | BIT_VALUES) <= budget + 1e-9
        score[~allowed] = -np.inf
        choice = score.argmax(axis=1)
        active = allowed.any(axis=1)
        masks[active] |= BIT_VALUES[choice[active]]
    return masks
