import numpy as np
import pandas as pd
import pytest

from t2d_aaf.research import crossfit as module
from t2d_aaf.survival.baseline import PREDICTOR_FEATURES
from t2d_aaf.survival.static_baselines import FittedStaticSubset


def cohort():
    frame = pd.DataFrame({name: np.arange(20, dtype=float) + 1 for name in PREDICTOR_FEATURES})
    frame['gender'] = np.tile([1., 2.], 10)
    frame['family_history_diabetes'] = np.tile([0., 1.], 10)
    frame['participant_id'] = np.arange(20)
    frame['split'] = 'train'
    frame['event'] = np.tile([0, 1], 10)
    frame['duration_years'] = np.tile([5., 2.], 10)
    frame.loc[0, ['age', 'gender']] = np.nan
    return frame


def test_folds_fit_preprocessing_only_on_training_participants(tmp_path, monkeypatch):
    frame = cohort()
    calls = []

    def fit(matrix, train, names, subset):
        calls.append(set(train.participant_id))
        return FittedStaticSubset(subset, tuple(names), np.zeros(len(names)),
                                   np.array([2.]), np.array([.1]), 0., ())

    monkeypatch.setattr(module, 'fit_static_subset', fit)
    result = module.crossfit(frame, cache=tmp_path, folds=2, workers=1)
    assert result['prepared'].participants == len(frame)
    assert len(calls) == 128
    for i, (train, held) in enumerate(module.fold_indices(frame, 2, 20260821)):
        assert calls[i * 64] == set(frame.iloc[train].participant_id)
        assert not calls[i * 64] & set(frame.iloc[held].participant_id)
        assert result['audit'][i]['continuous_imputation'][0] == frame.iloc[train].age.median()
        assert result['audit'][i]['participant_overlap'] == 0
    changed = frame.copy()
    changed.loc[1, 'age'] = 10000
    with pytest.raises(ValueError, match='cache'):
        module.crossfit(changed, cache=tmp_path, folds=2, workers=1)


def test_crossfit_rejects_test_rows_and_duplicate_participants():
    frame = cohort()
    frame.loc[0, 'split'] = 'test'
    with pytest.raises(ValueError, match='training split'):
        module.fold_indices(frame, 2, 1)
    frame['split'] = 'train'
    frame.loc[0, 'participant_id'] = frame.loc[1, 'participant_id']
    with pytest.raises(ValueError, match='unique'):
        module.fold_indices(frame, 2, 1)
