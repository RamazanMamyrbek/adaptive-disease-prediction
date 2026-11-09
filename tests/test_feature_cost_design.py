from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from t2d_aaf.data.cohort import (
    BASE_FEATURES,
    CORE_ACQUIRABLE_FEATURES,
    EXCLUDED_RAW_FIELDS,
    METADATA_COLUMNS,
    OPTIONAL_FEATURES,
    TARGET_COLUMNS,
)
from t2d_aaf.features.registry import load_cost_config, load_feature_cost_design, load_feature_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_CONFIG_PATH = PROJECT_ROOT / "configs" / "features.yaml"
COST_CONFIG_PATH = PROJECT_ROOT / "configs" / "costs.yaml"
COHORT_PATH = PROJECT_ROOT / "data" / "processed" / "primary_cohort.csv"
COHORT_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "primary_cohort_manifest.json"
FEATURE_COST_MANIFEST_PATH = PROJECT_ROOT / "reports" / "feature_cost_design_manifest.json"


def test_feature_registry_matches_confirmed_cohort_roles() -> None:
    registry = load_feature_registry(FEATURE_CONFIG_PATH)

    assert registry.names_for_role("base") == BASE_FEATURES
    assert registry.names_for_role("acquirable") == CORE_ACQUIRABLE_FEATURES
    assert registry.names_for_role("optional") == OPTIONAL_FEATURES
    assert registry.names_for_role("metadata") == METADATA_COLUMNS
    assert registry.names_for_role("target") == TARGET_COLUMNS
    assert {field.name for field in registry.raw_excluded_fields} == set(EXCLUDED_RAW_FIELDS)
    assert len(registry.cohort_feature_names) == 24
    assert len(set(registry.cohort_feature_names)) == 24


def test_cost_schemes_cover_only_core_candidates_and_use_relative_units() -> None:
    design = load_feature_cost_design(FEATURE_CONFIG_PATH, COST_CONFIG_PATH)
    costs = design.costs

    assert costs.default_scheme == "uniform_unit"
    assert costs.semantics["unit_type"] == "relative_experimental"
    assert costs.semantics["currency"] is None
    assert costs.semantics["real_prices"] is False
    assert costs.stop_cost == 0.0
    assert set(costs.base_feature_costs) == set(BASE_FEATURES)
    assert all(value == 0.0 for value in costs.base_feature_costs.values())
    for scheme in costs.schemes.values():
        assert set(scheme.costs) == set(CORE_ACQUIRABLE_FEATURES)
        assert all(value > 0.0 for value in scheme.costs.values())


def test_primary_and_fpg_sensitivity_cost_totals_are_explicit() -> None:
    costs = load_feature_cost_design(FEATURE_CONFIG_PATH, COST_CONFIG_PATH).costs

    assert costs.acquisition_cost(CORE_ACQUIRABLE_FEATURES) == pytest.approx(6.0)
    assert costs.acquisition_cost(
        CORE_ACQUIRABLE_FEATURES, scheme="fpg_discounted_sensitivity"
    ) == pytest.approx(5.5)
    assert costs.acquisition_cost(
        CORE_ACQUIRABLE_FEATURES, scheme="fpg_premium_sensitivity"
    ) == pytest.approx(7.0)
    assert costs.scheme("uniform_unit").costs["fpg"] == pytest.approx(1.0)
    assert costs.scheme("fpg_discounted_sensitivity").costs["fpg"] == pytest.approx(0.5)
    assert costs.scheme("fpg_premium_sensitivity").costs["fpg"] == pytest.approx(2.0)
    assert set(costs.scheme("uniform_unit").costs.values()) == {1.0}
    for feature in set(CORE_ACQUIRABLE_FEATURES) - {"fpg"}:
        assert costs.scheme("fpg_discounted_sensitivity").costs[feature] == pytest.approx(1.0)
        assert costs.scheme("fpg_premium_sensitivity").costs[feature] == pytest.approx(1.0)


def test_cost_api_rejects_non_candidates_and_duplicate_acquisition() -> None:
    costs = load_feature_cost_design(FEATURE_CONFIG_PATH, COST_CONFIG_PATH).costs

    with pytest.raises(KeyError, match="not acquirable"):
        costs.acquisition_cost(["age"])
    with pytest.raises(KeyError, match="not acquirable"):
        costs.acquisition_cost(["event"])
    with pytest.raises(KeyError, match="not acquirable"):
        costs.acquisition_cost(["FPG of final visit(mmol/L)"])
    with pytest.raises(ValueError, match="cannot be charged more than once"):
        costs.acquisition_cost(["fpg", "fpg"])


def test_negative_candidate_cost_is_rejected(tmp_path: Path) -> None:
    registry = load_feature_registry(FEATURE_CONFIG_PATH)
    document = yaml.safe_load(COST_CONFIG_PATH.read_text(encoding="utf-8"))
    invalid_document = deepcopy(document)
    invalid_document["schemes"]["uniform_unit"]["costs"]["bun"] = -1.0
    invalid_path = tmp_path / "invalid_costs.yaml"
    invalid_path.write_text(yaml.safe_dump(invalid_document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="strictly positive"):
        load_cost_config(invalid_path, registry)


def test_real_cohort_header_and_cohort_manifest_match_configs() -> None:
    if not COHORT_PATH.exists() or not COHORT_MANIFEST_PATH.exists():
        pytest.skip("Run scripts/build_cohort.py before validating generated data artifacts.")
    design = load_feature_cost_design(FEATURE_CONFIG_PATH, COST_CONFIG_PATH)
    with COHORT_PATH.open("r", encoding="utf-8", newline="") as file:
        header = next(csv.reader(file))
    cohort_manifest = json.loads(COHORT_MANIFEST_PATH.read_text(encoding="utf-8"))

    design.validate_cohort_columns(header)
    design.validate_cohort_manifest(cohort_manifest)


def test_generated_feature_cost_manifest_contains_no_model_or_policy_result() -> None:
    if not FEATURE_COST_MANIFEST_PATH.exists():
        pytest.skip("Run scripts/validate_feature_costs.py to generate the feature and cost design manifest.")
    manifest = json.loads(FEATURE_COST_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["stage"] == 4
    assert manifest["validation"]["cohort_header_matches_registry"] is True
    assert manifest["validation"]["real_prices_claimed"] is False
    assert manifest["model_training"] is None
    assert manifest["acquisition_policy"] is None
    assert set(manifest["roles"]["acquirable"]) == set(CORE_ACQUIRABLE_FEATURES)
