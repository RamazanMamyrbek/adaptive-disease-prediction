from __future__ import annotations

import json
from pathlib import Path

import pytest

from t2d_aaf.external.nagala import (
    COMPATIBLE_ACQUIRABLE_FEATURES,
    EXPECTED_EVENTS,
    EXPECTED_MD5,
    EXPECTED_ROWS,
    UNAVAILABLE_ACQUIRABLE_FEATURES,
    load_nagala_cohort,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_PATH = PROJECT_ROOT / "data" / "external" / "Data set Int J Obesity.xlsx"
METRICS_PATH = PROJECT_ROOT / "reports" / "external_validation_metrics.json"


def test_verified_nagala_mapping_preserves_rows_outcome_and_primary_gender_codes() -> None:
    if not EXTERNAL_PATH.exists():
        pytest.skip("Download the verified NAGALA workbook to data/external.")

    cohort = load_nagala_cohort(EXTERNAL_PATH)
    mapped = cohort.mapped

    assert len(mapped) == EXPECTED_ROWS
    assert int(mapped["event"].sum()) == EXPECTED_EVENTS
    assert mapped["participant_id"].is_unique
    assert mapped["duration_years"].gt(0).all()
    assert mapped["gender"].value_counts().to_dict() == {1.0: 8430, 2.0: 7034}
    assert mapped["family_history_diabetes"].isna().all()
    assert mapped.loc[:, UNAVAILABLE_ACQUIRABLE_FEATURES].isna().all().all()
    assert not mapped.loc[:, COMPATIBLE_ACQUIRABLE_FEATURES].isna().any().any()
    assert cohort.audit["md5"] == EXPECTED_MD5
    assert cohort.audit["integrity"]["rows_excluded"] == 0


def test_generated_external_results_preserve_external_validation_contract() -> None:
    if not METRICS_PATH.exists():
        pytest.skip("Run scripts/evaluate_external.py to generate external validation results.")
    raw_text = METRICS_PATH.read_text(encoding="utf-8")
    results = json.loads(raw_text)

    assert results["stage"] == 9
    assert results["scope"].startswith("partial compatible-feature")
    assert results["external_dataset"]["participants"] == EXPECTED_ROWS
    assert results["external_dataset"]["events"] == EXPECTED_EVENTS
    assert results["external_dataset"]["raw_file_committed"] is False
    assert results["evaluation"][
        "external_outcomes_used_for_selection_or_recalibration"
    ] is False
    assert results["model_or_policy_refit_on_external"] is False
    assert results["external_rows_excluded"] == 0
    assert results["participant_level_output"] is None
    assert set(results["mapping"]["compatible_acquirable_features"]) == set(
        COMPATIBLE_ACQUIRABLE_FEATURES
    )
    assert set(results["mapping"]["unavailable_acquirable_features"]) == set(
        UNAVAILABLE_ACQUIRABLE_FEATURES
    )
    assert set(results["evaluation"]["policy_metadata"]["available_adaptive_actions"]) == {
        *COMPATIBLE_ACQUIRABLE_FEATURES,
        "STOP",
    }
    assert results["bootstrap"]["replicates_requested"] == 200
    assert results["not_evaluated_exactly"]["full_features"]
    assert results["not_evaluated_exactly"]["random_acquisition"]
    assert "Infinity" not in raw_text
