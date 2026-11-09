import numpy as np
import pytest
from pathlib import Path
from sklearn.linear_model import Ridge

from t2d_aaf.acquisition.adaptive import PreparedAdaptiveData, build_state_matrix
from t2d_aaf.research.policy import (
    MarginalCosts, PANEL_MASKS, ScoreTable, predict_gain, simulate, utility_targets,
)


def test_marginal_costs_charge_draw_once_and_reflex_omits_it():
    costs = MarginalCosts(draw=10)
    assert costs.delta(0, 1) == 11
    assert costs.delta(1, 2) == 1
    assert costs.delta(0, 3) == 12
    assert costs.delta(3, 3) == 0
    assert MarginalCosts().delta(0, 3) == 2
    assert costs.delta(0, 1) + costs.delta(1, 2) == costs.total(3)


@pytest.mark.parametrize('draw', [-1, float('nan'), float('inf')])
def test_invalid_economic_parameters(draw):
    with pytest.raises(ValueError):
        MarginalCosts(draw=draw)


def test_brier_gain_keeps_harmful_changes_negative_and_censored_weight_zero():
    result = utility_targets(np.array([.2, .2, .2]), np.array([.4, .1, .9]), 'brier',
                             np.array([0, 0, 1]), np.array([2., 2., 0.]))
    np.testing.assert_allclose(result, [-.24, .06, 0.])
    np.testing.assert_array_equal(predict_gain(result, 'brier'), result)
    np.testing.assert_array_equal(predict_gain(np.array([-.5, 1.5]), 'revision'), [0., 1.])


def test_joint_start_can_overcome_myopic_stop_at_same_penalty():
    steps = np.zeros((2, 64, 6))
    steps[:, 0, :2] = .6
    panels = np.zeros((2, len(PANEL_MASKS)))
    panels[:, np.flatnonzero(PANEL_MASKS == 3)[0]] = 1.6
    scores = ScoreTable(steps, panels)
    costs = MarginalCosts(draw=1)
    np.testing.assert_array_equal(simulate(scores, costs, .5).masks, [0, 0])
    np.testing.assert_array_equal(simulate(scores, costs, .5, panels=True).masks, [3, 3])
    np.testing.assert_array_equal(simulate(scores, costs, .5, panels=True, available=1).masks, [0, 0])


def test_unavailable_and_already_acquired_features_cannot_be_requested():
    scores = ScoreTable(np.ones((3, 64, 6)), np.ones((3, len(PANEL_MASKS))))
    assert np.all(simulate(scores, MarginalCosts(), 0, available=15).masks == 15)
    assert np.all(simulate(scores, MarginalCosts(), np.inf).masks == 0)
    assert np.all(simulate(scores, MarginalCosts(), 1).masks == 0)


def test_hidden_values_and_missingness_do_not_enter_current_state():
    risks = np.zeros((2, 64))
    prepared = PreparedAdaptiveData(np.ones((2, 6)), np.array([[1] * 6, [99] * 6]),
                                    np.ones((2, 6)), np.array([[0] * 6, [1] * 6]), risks)
    state = build_state_matrix(prepared, np.zeros(2, dtype=int))
    np.testing.assert_array_equal(state[0], state[1])


def test_zero_draw_myopic_controller_reproduces_original_policy():
    from t2d_aaf.acquisition.adaptive import load_adaptive_model_bundle, prepare_adaptive_data, simulate_adaptive_policy
    from t2d_aaf.external.validation import load_adaptive_value_models
    from t2d_aaf.features.registry import load_feature_cost_design
    from t2d_aaf.survival.baseline import load_cohort_splits
    from t2d_aaf.research.policy import UtilityModels, score_table
    from threadpoolctl import threadpool_limits
    root = Path(__file__).resolve().parents[1]
    if not (root / 'models/adaptive_value_policy.joblib').exists():
        pytest.skip('Local research artifacts are not installed.')
    with threadpool_limits(limits=1):
        frame = load_cohort_splits(root / 'data/processed/primary_cohort.csv').test.iloc[:512]
        bundle = load_adaptive_model_bundle(root / 'reports/static_baselines_metrics.json', root / 'models/static_subsets.joblib')
        prepared = prepare_adaptive_data(bundle, frame)
        original = load_adaptive_value_models(root / 'models/adaptive_value_policy.joblib')
        panels = Ridge().fit(np.zeros((2, 31)), np.zeros((2, len(PANEL_MASKS))))
        model = UtilityModels('revision', tuple(item.estimator for item in original.models), panels, {})
        config = load_feature_cost_design(root / 'configs/features.yaml', root / 'configs/costs.yaml').costs
        penalty = .0010131473450483849
        expected = simulate_adaptive_policy(prepared, original, config, penalty)
        actual = simulate(score_table(prepared, model), MarginalCosts(), penalty)
        np.testing.assert_array_equal(actual.masks, expected.final_masks)
