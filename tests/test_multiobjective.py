import numpy as np
import pandas as pd
import pytest

from t2d_aaf.acquisition.adaptive import PreparedAdaptiveData
from t2d_aaf.clinical.thresholds import lab_labels
from t2d_aaf.clinical.rules import screening_triggers, apply_guardrail
from t2d_aaf.multiobjective.policy import budget_policy, fit_labs, lab_probabilities
from t2d_aaf.multiobjective.evaluation import contributions, summarize_counts, lab_intervals
from t2d_aaf.research.policy import MarginalCosts
from t2d_aaf.triage_experiment import select, prefix


def test_raw_labels_boundaries_missing_and_sex():
    frame = pd.DataFrame({'gender': [1, 2, np.nan, 1], 'total_cholesterol': [5.2, 5.1, np.nan, 6.],
                          'triglycerides': [1.7, 1.6, 3., np.nan], 'alt': [33., 25.01, 100., np.nan],
                          'bun': [7.1, 7.2, np.nan, 8.]})
    labels = lab_labels(frame)
    np.testing.assert_allclose(labels[0], [1, 1, 0, 0])
    np.testing.assert_allclose(labels[1], [0, 0, 1, 1])
    assert np.isnan(labels[2, 2]) and np.isnan(labels[3, 1])
    assert np.isnan(lab_labels(frame, 15)[:, 3]).all()


def test_guidelines_are_separate_and_do_not_impute_risk_factors():
    frame = pd.DataFrame({'age': [35, 45, 46, 30, 30, np.nan], 'bmi': [20, 20, 20, 23, 25, np.nan],
                          'sbp': [100, 100, 100, 130, 139, np.nan], 'dbp': [60, 60, 60, 79, 89, np.nan],
                          'family_history_diabetes': [1] * 6})
    ada = screening_triggers(frame, 'ADA2026')['require_fpg']
    kz = screening_triggers(frame, 'KZ158')['require_fpg']
    np.testing.assert_array_equal(ada, [True, True, True, False, True, False])
    np.testing.assert_array_equal(kz, [False, False, True, False, False, False])
    assert screening_triggers(frame, asian_ancestry=True)['require_fpg'][3]
    assert not screening_triggers(frame, 'KZ158', asian_ancestry=True)['require_fpg'][4]
    assert screening_triggers(frame, 'KZ158', high_risk_ethnicity=np.ones(6))['require_fpg'][4]
    assert screening_triggers(frame, 'KZ158', first_degree_family=np.ones(6))['require_fpg'][4]
    assert not screening_triggers(frame, 'KZ158')['require_fpg'][4]
    with pytest.raises(ValueError):
        screening_triggers(frame, 'unknown')


def test_hybrid_cost_adds_fpg_and_charges_draw_once():
    before = np.array([0, 2, 1, 3])
    after = apply_guardrail(before, {'require_fpg': np.ones(4, bool)})
    np.testing.assert_array_equal(after, [1, 3, 1, 3])
    costs = MarginalCosts(draw=20)
    np.testing.assert_array_equal(costs.total(after) - costs.total(before), [21, 1, 0, 0])


def test_budget_selection_masks_unavailable_and_unaffordable_tests():
    revision = np.zeros((2, 64, 6))
    discovery = np.zeros_like(revision)
    discovery[:, :, 1] = 1
    discovery[:, :, 5] = 100
    costs = MarginalCosts(draw=20)
    np.testing.assert_array_equal(budget_policy(revision, discovery, 1, 20, costs), [0, 0])
    np.testing.assert_array_equal(budget_policy(revision, discovery, 1, 21, costs, available=15), [2, 2])
    masks = budget_policy(revision, discovery, 1, 22, costs, available=15)
    assert all(int(mask).bit_count() == 2 for mask in masks)
    assert np.all(costs.total(masks) <= 22)
    assert prefix([32, 1, 2, 4, 8, 16], 3, 15) == 7


def test_missing_labels_never_create_false_positive_or_negative_truth():
    labels = np.array([[1, np.nan, 0, np.nan], [0, 1, np.nan, np.nan]])
    masks = np.array([63, 1])
    total = contributions(labels, masks).sum(axis=0)
    summary = summarize_counts(total, 2, 7)
    assert summary['lab_micro_recall'] == .5
    assert summary['lab_missed_fraction'] == .5
    assert summary['lab_yield_per_participant'] == .5
    assert summary['lab_unknown_acquisitions_per_participant'] == 1.
    assert np.isnan(summary['bun_recall'])


def test_paired_lab_intervals_zero_for_identical_routes():
    labels = np.tile([1, 0, 1, np.nan], (8, 1))
    frame = pd.DataFrame({'event': [0, 1] * 4})
    masks = np.full(8, 15)
    result = lab_intervals(frame, labels, {'a': masks, 'b': masks}, [('a', 'b')], replicates=20)
    for metric, interval in result['contrasts']['a_minus_b'].items():
        if interval['point_estimate'] is not None:
            assert interval['point_estimate'] == interval['ci95_lower'] == interval['ci95_upper'] == 0
    assert result['methods']['a']['bun_recall']['valid_replicates'] == 0


def test_validation_infeasibility_is_explicit():
    curve = [{'weight': 0, 'c_index': .8, 'lab_recall': .1}, {'weight': 1, 'c_index': .79, 'lab_recall': .9}]
    assert select(curve, .81)['retention_constraint_met'] is False
    assert select(curve, .81)['weight'] == 0
    assert select(curve, .78)['weight'] == 1


def test_unrevealed_own_laboratory_values_do_not_change_probabilities():
    rng = np.random.default_rng(7)
    n = 80
    base = rng.normal(size=(n, 6))
    labs = rng.normal(size=(n, 6))
    risks = np.tile(np.linspace(.01, .04, 64), (n, 1))
    prepared = PreparedAdaptiveData(base, labs, np.ones_like(base), np.ones_like(labs), risks)
    labels = (rng.random((n, 4)) > .5).astype(float)
    labels[:10, 3] = np.nan
    model = fit_labs(prepared, labels)
    before, availability = lab_probabilities(prepared, np.zeros(n, int), model)
    changed = PreparedAdaptiveData(base, labs * 1000, np.ones_like(base), np.zeros_like(labs), risks)
    after, observed = lab_probabilities(changed, np.zeros(n, int), model)
    np.testing.assert_allclose(before, after)
    np.testing.assert_allclose(availability, observed)
    assert model.summary[3]['observed_participants'] == 70
