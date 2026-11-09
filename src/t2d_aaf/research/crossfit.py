from __future__ import annotations

from dataclasses import fields
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from t2d_aaf.acquisition.adaptive import AdaptiveModelBundle, PreparedAdaptiveData, prepare_adaptive_data
from t2d_aaf.evaluation.cost_performance import fixed_horizon_ipcw_outcome
from t2d_aaf.survival.baseline import PREDICTOR_FEATURES, build_preprocessor
from t2d_aaf.survival.static_baselines import enumerate_candidate_subsets, fit_static_subset


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def atomic_dump(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    joblib.dump(value, temporary)
    temporary.replace(path)


def fold_indices(frame: pd.DataFrame, folds: int, seed: int):
    if frame['participant_id'].isna().any() or frame['participant_id'].duplicated().any():
        raise ValueError('OOF requires unique participant identifiers.')
    if set(frame['split']) != {'train'}:
        raise ValueError('Only the training split may enter cross-fitting.')
    if not 2 <= folds <= int(frame.groupby('event').size().min()):
        raise ValueError('Insufficient events or invalid fold count.')
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return tuple(splitter.split(frame, frame['event']))


def _fit_fold(frame, train_rows, held_rows, index, cache, horizon):
    folder = cache / f'fold_{index}'
    folder.mkdir(parents=True, exist_ok=True)
    completed = folder / 'predictions.joblib'
    if completed.exists():
        print(f'OOF fold {index}: restored checkpoint', flush=True)
        return joblib.load(completed)
    train = frame.iloc[train_rows]
    held = frame.iloc[held_rows]
    with threadpool_limits(limits=1):
        processor = build_preprocessor()
        matrix = np.asarray(processor.fit_transform(train.loc[:, PREDICTOR_FEATURES]), dtype=float)
        names = tuple(processor.get_feature_names_out())
        models = {}
        for number, subset in enumerate(enumerate_candidate_subsets()):
            path = folder / f'subset_{number:02d}.joblib'
            if path.exists():
                model = joblib.load(path)
            else:
                model = fit_static_subset(matrix, train, names, subset)
                if any('converg' in message.lower() for message in model.fit_warnings):
                    raise RuntimeError(f'Cox convergence warning: fold={index}, subset={subset}')
                atomic_dump(model, path)
            models[subset] = model
            if (number + 1) % 8 == 0:
                print(f'OOF fold {index}: {number + 1}/64 Cox models', flush=True)
        bundle = AdaptiveModelBundle(processor, names, models, {})
        prepared = prepare_adaptive_data(bundle, held, horizon=horizon)
        outcome = fixed_horizon_ipcw_outcome(
            train.duration_years, train.event, held.duration_years, held.event, horizon,
        )
        audit = {
            'fold': index, 'training_participants': len(train), 'held_out_participants': len(held),
            'participant_overlap': len(set(train.participant_id) & set(held.participant_id)),
            'training_rows_sha256': hashlib.sha256(train_rows.tobytes()).hexdigest(),
            'held_out_rows_sha256': hashlib.sha256(held_rows.tobytes()).hexdigest(),
            'continuous_imputation': processor.named_transformers_['continuous'].named_steps['imputer'].statistics_.tolist(),
            'continuous_mean': processor.named_transformers_['continuous'].named_steps['scaler'].mean_.tolist(),
            'binary_imputation': processor.named_transformers_['binary'].named_steps['imputer'].statistics_.tolist(),
            'fit_warnings': {str(key): list(value.fit_warnings) for key, value in models.items() if value.fit_warnings},
        }
        result = (held_rows, prepared, outcome.labels, outcome.weights, audit)
        atomic_dump(bundle, folder / 'bank.joblib')
        atomic_dump(result, completed)
        return result


def crossfit(frame, *, cache: Path, folds=5, seed=20260821, horizon=3.0, workers=2):
    cache.mkdir(parents=True, exist_ok=True)
    from t2d_aaf.survival import baseline, static_baselines
    from t2d_aaf.acquisition import adaptive
    from t2d_aaf.evaluation import cost_performance
    signature = {
        'data': hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        'folds': folds, 'seed': seed, 'horizon': horizon,
        'code': {Path(module.__file__).name: digest(Path(module.__file__))
                 for module in (baseline, static_baselines, adaptive, cost_performance)},
        'crossfit_code': digest(Path(__file__)),
    }
    signature_path = cache / 'signature.joblib'
    if signature_path.exists() and joblib.load(signature_path) != signature:
        raise ValueError('OOF cache belongs to different data, code or settings; use a new directory.')
    atomic_dump(signature, signature_path)
    partitions = fold_indices(frame, folds, seed)
    results = Parallel(n_jobs=workers)(
        delayed(_fit_fold)(frame, train, held, i, cache, horizon)
        for i, (train, held) in enumerate(partitions)
    )
    arrays = {field.name: np.empty((len(frame), getattr(results[0][1], field.name).shape[1]))
              for field in fields(PreparedAdaptiveData)}
    labels, weights = np.empty(len(frame), dtype=int), np.empty(len(frame))
    coverage = np.zeros(len(frame), dtype=int)
    for rows, prepared, y, w, audit in results:
        for name in arrays:
            arrays[name][rows] = getattr(prepared, name)
        labels[rows], weights[rows] = y, w
        coverage[rows] += 1
        if audit['participant_overlap']:
            raise AssertionError('Training and held-out participants overlap.')
    if not np.all(coverage == 1):
        raise AssertionError('Every participant must receive exactly one OOF prediction.')
    result = {'prepared': PreparedAdaptiveData(**arrays), 'labels': labels, 'weights': weights,
              'audit': [item[4] for item in results], 'signature': signature}
    atomic_dump(result, cache / 'oof.joblib')
    return result
