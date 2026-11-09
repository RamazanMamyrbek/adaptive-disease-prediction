from __future__ import annotations

import hashlib

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from t2d_aaf.evaluation.cost_performance import fixed_horizon_ipcw_outcome
from t2d_aaf.evaluation.survival_metrics import harrell_c_index


METRICS = ('harrell_c_index', 'ipcw_brier_at_horizon', 'ipcw_cumulative_dynamic_auc',
           'ipcw_average_precision', 'mean_cost', 'mean_tests', 'stop_without_tests_fraction')


def _prediction_metrics(duration, event, risk, labels, weights, included):
    return [harrell_c_index(duration, event, risk),
            float(np.mean(weights * (labels - risk) ** 2)),
            float(roc_auc_score(labels[included], risk[included], sample_weight=weights[included])),
            float(average_precision_score(labels[included], risk[included], sample_weight=weights[included]))]


def _replicate(seed, duration, event, risks, labels, weights, included, costs, counts):
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(seed)
        rows = np.concatenate([rng.choice(group, len(group), replace=True)
                               for group in (np.flatnonzero(event == 0), np.flatnonzero(event == 1))])
        predictive = np.array([_prediction_metrics(duration[rows], event[rows], risk[rows],
                               labels[rows], weights[rows], included[rows]) for risk in risks])
        economic = np.column_stack([costs[:, rows].mean(axis=1), counts[:, rows].mean(axis=1),
                                    (counts[:, rows] == 0).mean(axis=1)])
        return predictive, economic


def paired_intervals(train, frame, risks, costs, counts, contrasts, *, horizon=3., replicates=200,
                     seed=20260827, workers=4):
    if replicates < 2:
        raise ValueError('At least two bootstrap replicates are required.')
    names = list(risks)
    if set(names) != set(costs) or set(names) != set(counts):
        raise ValueError('Prediction, cost and test-count methods must match.')
    duration, event = frame.duration_years.to_numpy(float), frame.event.to_numpy(int)
    outcome = fixed_horizon_ipcw_outcome(train.duration_years, train.event, duration, event, horizon)
    unique, lookup, indices = [], {}, []
    for name in names:
        risk = np.asarray(risks[name], dtype=float)
        key = hashlib.sha256(risk.tobytes()).hexdigest()
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append(risk)
        indices.append(lookup[key])
    risk_matrix = np.stack(unique)
    cost_matrix = np.stack([costs[name] for name in names])
    count_matrix = np.stack([counts[name] for name in names])
    if risk_matrix.shape[1] != len(frame) or not np.isfinite(risk_matrix).all():
        raise ValueError('Invalid risk arrays.')
    if any(left not in names or right not in names for left, right in contrasts):
        raise ValueError('Unknown contrast method.')
    seeds = np.random.SeedSequence(seed).spawn(replicates)
    results = Parallel(n_jobs=workers, max_nbytes='1M', verbose=5)(
        delayed(_replicate)(s, duration, event, risk_matrix, outcome.labels, outcome.weights,
                            outcome.included, cost_matrix, count_matrix) for s in seeds
    )
    samples = np.stack([np.column_stack((prediction[indices], economic)) for prediction, economic in results])
    predictive = np.array([_prediction_metrics(duration, event, risk, outcome.labels, outcome.weights,
                                               outcome.included) for risk in unique])
    points = np.column_stack((predictive[indices], cost_matrix.mean(axis=1), count_matrix.mean(axis=1),
                              (count_matrix == 0).mean(axis=1)))

    def summarize(values, point):
        return {metric: {'point_estimate': float(point[index]),
                         'ci95_lower': float(np.quantile(values[:, index], .025)),
                         'ci95_upper': float(np.quantile(values[:, index], .975)),
                         'valid_replicates': replicates}
                for index, metric in enumerate(METRICS)}

    return {
        'method': 'event-stratified paired percentile bootstrap, conditional on fitted policies and censoring weights',
        'replicates': replicates, 'seed': seed, 'unique_prediction_vectors': len(unique),
        'censoring_refit_per_replicate': False, 'model_refit_per_replicate': False,
        'multiplicity_adjusted': False,
        'methods': {name: summarize(samples[:, i], points[i]) for i, name in enumerate(names)},
        'contrasts': {left + '_minus_' + right: summarize(samples[:, names.index(left)] - samples[:, names.index(right)],
                                                        points[names.index(left)] - points[names.index(right)])
                      for left, right in contrasts},
    }
