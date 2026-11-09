from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from t2d_aaf.data.cohort import (
    BASE_FEATURES,
    CORE_ACQUIRABLE_FEATURES,
    EXCLUDED_FROM_MAIN_PREDICTORS,
)
from t2d_aaf.survival.static_baselines import (
    canonical_subset,
    derive_static_orders,
    enumerate_candidate_subsets,
    generate_random_orders,
    prefix_subsets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "reports" / "static_baselines_metrics.json"


def _synthetic_evaluations() -> dict[tuple[str, ...], dict[str, dict[str, float]]]:
    evaluations = {}
    weights = {feature: 0.01 * (index + 1) for index, feature in enumerate(CORE_ACQUIRABLE_FEATURES)}
    for subset in enumerate_candidate_subsets():
        score = 0.60 + sum(weights[feature] for feature in subset)
        evaluations[subset] = {
            "validation": {"harrell_c_index": score},
            "test": {"harrell_c_index": 1.0 - score},
        }
    return evaluations


def test_all_64_candidate_subsets_are_enumerated_once() -> None:
    subsets = enumerate_candidate_subsets()

    assert len(subsets) == 2 ** len(CORE_ACQUIRABLE_FEATURES) == 64
    assert len(set(subsets)) == 64
    assert subsets[0] == ()
    assert subsets[-1] == CORE_ACQUIRABLE_FEATURES


def test_static_predictor_contract_excludes_leakage_fields() -> None:
    allowed = set(BASE_FEATURES) | set(CORE_ACQUIRABLE_FEATURES)

    assert allowed.isdisjoint(EXCLUDED_FROM_MAIN_PREDICTORS)
    assert allowed.isdisjoint({"participant_id", "site", "event", "duration_years"})


def test_top_k_and_greedy_orders_do_not_read_test_metrics() -> None:
    original = _synthetic_evaluations()
    changed_test = deepcopy(original)
    for result in changed_test.values():
        result["test"]["harrell_c_index"] = -999.0

    assert derive_static_orders(original) == derive_static_orders(changed_test)


def test_random_baseline_is_reproducible_and_data_independent() -> None:
    first = generate_random_orders(repeats=25, seed=17)
    second = generate_random_orders(repeats=25, seed=17)
    different = generate_random_orders(repeats=25, seed=18)

    assert first == second
    assert first != different
    assert all(set(order) == set(CORE_ACQUIRABLE_FEATURES) for order in first)


def test_prefixes_lookup_canonical_subsets_without_changing_global_order() -> None:
    order = tuple(reversed(CORE_ACQUIRABLE_FEATURES))
    prefixes = prefix_subsets(order)

    assert len(prefixes) == len(CORE_ACQUIRABLE_FEATURES) + 1
    assert prefixes[0] == ()
    assert prefixes[1] == canonical_subset((order[0],))
    assert prefixes[-1] == CORE_ACQUIRABLE_FEATURES


def test_generated_static_results_preserve_static_and_leakage_contract() -> None:
    if not METRICS_PATH.exists():
        pytest.skip("Run scripts/run_static_baselines.py to generate static baseline results.")
    results = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    assert results["stage"] == 5
    assert results["model"]["fit_split"] == "train"
    assert results["model"]["optimizer"] == "bfgs"
    assert results["model"]["models_fitted"] == 64
    assert results["evaluation"]["selection_metric"] == "validation Harrell C-index"
    assert results["evaluation"]["test_used_for_selection"] is False
    assert results["costs"]["default_scheme"] == "uniform_unit"
    assert results["acquisition_policy"] is None
    assert len(results["all_subset_results"]) == 64
    assert set(results["methods"]) == {
        "base_only",
        "full_features",
        "static_top_k",
        "random",
        "importance_greedy",
    }
