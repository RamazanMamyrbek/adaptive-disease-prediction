from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from t2d_aaf.acquisition.adaptive import prepare_adaptive_data
from t2d_aaf.clinical.thresholds import LABS, lab_labels
from t2d_aaf.multiobjective.policy import fit_labs, lab_probabilities
from t2d_aaf.research.crossfit import atomic_dump, digest, fold_indices


def train(root: Path, frame, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    paths = [root / 'data/processed/primary_cohort.csv', root / 'models/research/oof/oof.joblib']
    paths += sorted((root / 'src/t2d_aaf/multiobjective').glob('*.py'))
    paths += sorted((root / 'src/t2d_aaf/clinical').glob('*.py'))
    signature = {str(path.relative_to(root)): digest(path) for path in paths}
    identity = cache / 'training_signature.joblib'
    if identity.exists() and joblib.load(identity) != signature:
        raise ValueError('Training inputs changed; choose a fresh model directory.')
    atomic_dump(signature, identity)
    completed = cache / 'labs.joblib'
    if completed.exists():
        return joblib.load(completed), joblib.load(cache / 'audit.joblib')
    labels = lab_labels(frame)
    folds = fold_indices(frame, 5, 20260821)
    probabilities = np.full_like(labels, np.nan)
    availability = np.full_like(labels, np.nan)
    audit = []
    with threadpool_limits(limits=1):
        for index, (training, held) in enumerate(folds):
            path = cache / f'fold_{index}.joblib'
            if path.exists():
                prediction, observed, record = joblib.load(path)
            else:
                print(f'Lab OOF fold {index + 1}/5: fold-local states and observed labels', flush=True)
                bundle = joblib.load(root / f'models/research/oof/fold_{index}/bank.joblib')
                expected, prepared_held, _, _, old_audit = joblib.load(
                    root / f'models/research/oof/fold_{index}/predictions.joblib')
                np.testing.assert_array_equal(held, expected)
                training_hash = hashlib.sha256(training.tobytes()).hexdigest()
                if training_hash != old_audit['training_rows_sha256']:
                    raise ValueError('Existing Cox fold does not match lab training fold.')
                prepared_train = prepare_adaptive_data(bundle, frame.iloc[training])
                models = fit_labs(prepared_train, labels[training], seed=20261110 + index)
                prediction, observed = lab_probabilities(prepared_held, np.zeros(len(held), int), models)
                record = {'fold': index, 'train_count': len(training), 'held_count': len(held),
                          'participant_overlap': int(len(np.intersect1d(training, held))),
                          'training_rows_sha256': training_hash,
                          'cox_bank_sha256': digest(root / f'models/research/oof/fold_{index}/bank.joblib'),
                          'lab_training': models.summary}
                atomic_dump((prediction, observed, record), path)
                del prepared_train, bundle, models
            probabilities[held], availability[held] = prediction, observed
            audit.append(record)
        oof = joblib.load(root / 'models/research/oof/oof.joblib')
        models = fit_labs(oof['prepared'], labels)
    diagnostics = {}
    for index, lab in enumerate(LABS):
        known = np.isfinite(labels[:, index])
        y, p = labels[known, index], probabilities[known, index]
        diagnostics[lab] = {'oof_base_brier_observed_only': float(brier_score_loss(y, p)),
                            'oof_base_auc_observed_only': float(roc_auc_score(y, p)),
                            'observed_count': int(known.sum()), 'positive_count': int(y.sum()),
                            'oof_availability_brier': float(brier_score_loss(known, availability[:, index]))}
    report = {'folds': audit, 'diagnostics': diagnostics, 'final_training': models.summary,
              'labels': 'Raw observed laboratory thresholds; missing labels excluded from abnormality loss',
              'utility': 'P(source observed | state) * P(abnormal | source observed, state)',
              'interpretation': 'Expected retrospective observable yield, not prospective test sensitivity',
              'final_state_source': 'Existing cross-fitted Cox states; preprocessing fitted within folds'}
    atomic_dump(report, cache / 'audit.joblib')
    atomic_dump(models, completed)
    return models, report
