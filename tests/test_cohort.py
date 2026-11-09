from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from t2d_aaf.data.cohort import (
    BASE_FEATURES,
    CORE_ACQUIRABLE_FEATURES,
    DEFAULT_SEED,
    EXCLUDED_FROM_MAIN_PREDICTORS,
    OPTIONAL_FEATURES,
    RAW_DURATION_COLUMN,
    RAW_EVENT_COLUMN,
    RAW_ID_COLUMN,
    clean_raw_record,
    split_participant_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COHORT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "primary_cohort.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "primary_cohort_manifest.json"


def test_event_and_duration_validation_accepts_expected_raw_fields() -> None:
    raw_record = {
        RAW_ID_COLUMN: 101,
        RAW_EVENT_COLUMN: 1,
        RAW_DURATION_COLUMN: 3.25,
        "Age (y)": 50,
        "Gender(1, male; 2, female)": 1,
        "site": 5,
        "BMI(kg/m2)": 24.5,
        "SBP(mmHg)": 120,
        "DBP(mmHg)": 75,
        "family histroy of diabetes(1,Yes;0,No)": 0,
        "FPG (mmol/L)": 5.2,
        "Cholesterol(mmol/L)": 4.8,
        "Triglyceride(mmol/L)": 1.1,
        "ALT(U/L)": 20,
        "CCR(umol/L)": 70,
        "BUN(mmol/L)": 4.6,
    }

    record, reasons = clean_raw_record(raw_record)

    assert reasons == []
    assert record is not None
    assert record["event"] == 1
    assert record["duration_years"] == 3.25


@pytest.mark.parametrize(
    ("event", "duration", "expected_reason"),
    [
        (2, 3.0, "invalid_event"),
        (1, 0.0, "invalid_duration"),
        (1, None, "invalid_duration"),
    ],
)
def test_event_and_duration_validation_rejects_invalid_values(
    event: int | None, duration: float | None, expected_reason: str
) -> None:
    raw_record = {
        RAW_ID_COLUMN: 101,
        RAW_EVENT_COLUMN: event,
        RAW_DURATION_COLUMN: duration,
    }

    record, reasons = clean_raw_record(raw_record)

    assert record is None
    assert expected_reason in reasons


def test_main_predictor_sets_exclude_leakage_targets_and_site() -> None:
    main_predictors = set(BASE_FEATURES) | set(CORE_ACQUIRABLE_FEATURES)

    assert main_predictors.isdisjoint(EXCLUDED_FROM_MAIN_PREDICTORS)
    assert "site" not in main_predictors
    assert "height_cm" not in main_predictors
    assert "weight_kg" not in main_predictors
    assert "smoking_status" not in main_predictors
    assert "drinking_status" not in main_predictors
    assert "hdl_c" not in main_predictors
    assert "ldl" not in main_predictors
    assert "ast" not in main_predictors


def test_stratified_split_is_reproducible_and_disjoint() -> None:
    records = [
        {"participant_id": participant_id, "event": 0}
        for participant_id in range(1, 101)
    ] + [
        {"participant_id": participant_id, "event": 1}
        for participant_id in range(101, 121)
    ]

    first = split_participant_ids(records, seed=DEFAULT_SEED)
    second = split_participant_ids(records, seed=DEFAULT_SEED)

    assert first == second
    assert set(first) == {record["participant_id"] for record in records}
    assert set(first.values()) == {"train", "validation", "test"}
    for participant_id in first:
        assigned_splits = {split for pid, split in first.items() if pid == participant_id}
        assert len(assigned_splits) == 1


def test_optional_features_are_not_core_predictors() -> None:
    assert set(OPTIONAL_FEATURES).isdisjoint(BASE_FEATURES)
    assert set(OPTIONAL_FEATURES).isdisjoint(CORE_ACQUIRABLE_FEATURES)


def test_generated_manifest_has_cohort_target_split_and_leakage_contract() -> None:
    if not MANIFEST_PATH.exists():
        pytest.skip("Generated cohort construction manifest is not present; run scripts/build_cohort.py first.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    main_predictors = set(manifest["base_features"]) | set(manifest["core_acquirable_features"])

    assert manifest["target_columns"] == ["event", "duration_years"]
    assert manifest["seed"] == DEFAULT_SEED
    assert main_predictors.isdisjoint(manifest["excluded_from_main_predictors"])
    assert "site" not in main_predictors
    assert "FPG of final visit(mmol/L)" in manifest["excluded_raw_fields"]
    assert "Diabetes diagnosed during followup（1,Yes）" in manifest["excluded_raw_fields"]


def test_generated_cohort_has_disjoint_splits_and_no_leakage_columns() -> None:
    if not COHORT_CSV_PATH.exists():
        pytest.skip("Generated cohort construction cohort is not present; run scripts/build_cohort.py first.")

    split_by_id: dict[str, str] = {}
    with COHORT_CSV_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames is not None
        assert "FPG of final visit(mmol/L)" not in reader.fieldnames
        assert "Diabetes diagnosed during followup（1,Yes）" not in reader.fieldnames
        assert "event" in reader.fieldnames
        assert "duration_years" in reader.fieldnames
        for row in reader:
            participant_id = row["participant_id"]
            split = row["split"]
            event = int(row["event"])
            duration = float(row["duration_years"])
            assert event in (0, 1)
            assert duration > 0
            if participant_id in split_by_id:
                assert split_by_id[participant_id] == split
            split_by_id[participant_id] = split

    assert set(split_by_id.values()) == {"train", "validation", "test"}
