from __future__ import annotations

import numpy as np

from t2d_aaf.clinical.thresholds import LABS, LAB_BITS


def contributions(labels, masks):
    acquired = (np.asarray(masks)[:, None] & LAB_BITS) != 0
    known = np.isfinite(labels)
    positive = labels == 1
    detected = positive & acquired
    return np.column_stack((positive, detected, known & acquired, ~known & acquired,
                            (detected.any(axis=1)), (positive.any(axis=1))))


def summarize_counts(totals, n, tests):
    positives, detected, measured, unknown = [totals[i:i + 4] for i in range(0, 16, 4)]

    def ratio(a, b):
        return float(a / b) if b else np.nan

    result = {'lab_micro_recall': ratio(detected.sum(), positives.sum()),
              'lab_missed_fraction': ratio((positives - detected).sum(), positives.sum()),
              'lab_yield_per_participant': ratio(detected.sum(), n),
              'lab_yield_per_paid_test': ratio(detected.sum(), tests),
              'lab_unknown_acquisitions_per_participant': ratio(unknown.sum(), n),
              'lab_any_patient_recall': ratio(totals[16], totals[17])}
    for i, name in enumerate(LABS):
        result[f'{name}_recall'] = ratio(detected[i], positives[i])
        result[f'{name}_yield_per_participant'] = ratio(detected[i], n)
        result[f'{name}_observed_acquisitions_per_participant'] = ratio(measured[i], n)
    return result


def lab_intervals(frame, labels, masks_by_method, contrasts, replicates=200, seed=20260827):
    names = list(masks_by_method)
    values = [contributions(labels, masks_by_method[name]) for name in names]
    counts = np.stack([np.array([int(m).bit_count() for m in masks_by_method[name]]) for name in names])
    points = [summarize_counts(value.sum(axis=0), len(frame), counts[i].sum()) for i, value in enumerate(values)]
    metrics = list(points[0])
    samples = np.zeros((replicates, len(names), len(metrics)))
    event = frame.event.to_numpy(int)
    for b, child in enumerate(np.random.SeedSequence(seed).spawn(replicates)):
        rng = np.random.default_rng(child)
        rows = np.concatenate([rng.choice(group, len(group), replace=True)
                               for group in (np.flatnonzero(event == 0), np.flatnonzero(event == 1))])
        for i, value in enumerate(values):
            summary = summarize_counts(value[rows].sum(axis=0), len(rows), counts[i, rows].sum())
            samples[b, i] = list(summary.values())

    def interval(vector, point):
        finite = vector[np.isfinite(vector)]
        return {'point_estimate': float(point) if np.isfinite(point) else None,
                'ci95_lower': float(np.quantile(finite, .025)) if len(finite) else None,
                'ci95_upper': float(np.quantile(finite, .975)) if len(finite) else None,
                'valid_replicates': len(finite)}

    output = {'methods': {}, 'contrasts': {}}
    for i, name in enumerate(names):
        output['methods'][name] = {metric: interval(samples[:, i, j], points[i][metric])
                                   for j, metric in enumerate(metrics)}
    for left, right in contrasts:
        a, b = names.index(left), names.index(right)
        output['contrasts'][left + '_minus_' + right] = {
            metric: interval(samples[:, a, j] - samples[:, b, j], points[a][metric] - points[b][metric])
            for j, metric in enumerate(metrics)}
    return output
