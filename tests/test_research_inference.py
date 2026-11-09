import numpy as np
import pandas as pd

from t2d_aaf.research.experiment import choose_point, ordered_masks, prefix_masks
from t2d_aaf.research.inference import paired_intervals
from t2d_aaf.acquisition.adaptive import PreparedAdaptiveData


def test_identical_methods_have_zero_paired_difference_and_cost_difference_is_exact():
    frame = pd.DataFrame({'duration_years': np.tile([1., 2., 4., 5.], 20),
                          'event': np.tile([1, 1, 0, 0], 20)})
    risk = np.linspace(.01, .8, len(frame))
    result = paired_intervals(frame, frame, {'a': risk, 'b': risk.copy()},
                              {'a': np.ones(len(frame)), 'b': np.full(len(frame), 3.)},
                              {'a': np.ones(len(frame)), 'b': np.ones(len(frame))}, [('a', 'b')],
                              replicates=8, workers=1)
    assert result['unique_prediction_vectors'] == 1
    for metric, interval in result['contrasts']['a_minus_b'].items():
        expected = -2. if metric == 'mean_cost' else 0.
        assert interval['point_estimate'] == expected
        assert interval['ci95_lower'] == expected
        assert interval['ci95_upper'] == expected


def test_selection_uses_constraint_and_marks_infeasible_policies():
    points = [{'parameter': 1., 'validation_c_index': .8, 'mean_cost': 1.},
              {'parameter': 0., 'validation_c_index': .9, 'mean_cost': 3.}]
    result = choose_point(points, .85)
    assert result['parameter'] == 0.
    assert result['retention_constraint_met']
    assert not choose_point(points, .95)['retention_constraint_met']


def test_external_order_intersects_available_features_before_replay():
    order = ('ccr', 'fpg', 'bun', 'alt', 'triglycerides', 'total_cholesterol')
    np.testing.assert_array_equal(prefix_masks(order, 15), [0, 1, 9, 13, 15])
    prepared = PreparedAdaptiveData(np.zeros((2, 6)), np.zeros((2, 6)), np.ones((2, 6)),
                                    np.ones((2, 6)), np.full((2, 64), .2))
    np.testing.assert_array_equal(ordered_masks(prepared, order, 0, 15), [15, 15])
