from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from t2d_aaf.data.cohort import BASE_FEATURES, CORE_ACQUIRABLE_FEATURES
from t2d_aaf.survival.baseline import PREDICTOR_FEATURES


DATASET_DOI = "10.5061/dryad.8q0p192"
EXPECTED_FILE_NAME = "Data set Int J Obesity.xlsx"
EXPECTED_SHEET = "Sheet 1"
EXPECTED_ROWS = 15_464
EXPECTED_COLUMNS = 31
EXPECTED_EVENTS = 373
EXPECTED_SIZE_BYTES = 3_255_571
EXPECTED_MD5 = "47edfb147078657a0eecfff7f9364532"
DAYS_PER_YEAR = 365.25

RAW_COLUMNS: tuple[str, ...] = (
    "Patient Number",
    "Sex, 1:Men, 2:Women",
    "Baseline Age, yrs",
    "Baseline Ethanol consumption g/wk",
    "Baseline Fatty liver 0/1",
    "Baseline BMI, kg/m2",
    "Baseline Waist circumference(WC), cm",
    "Baseline WC≥90 in Men, ≥80 in Women 0/1",
    "Baseline BMI≥25 0/1",
    "Obesity phenotype 1-8",
    "Follow up duration, days",
    "Baseline ALT, IU/L",
    "Baseline AST, IU/L",
    "Baseline Body Weight, kg",
    "Baseline Habit of exercise 0/1",
    "Baseline GGT, IU/L",
    "Baseline HDL-cholesterol, mg/dL",
    "Baseline HDL-cholesterol, mmol/L",
    "Baseline Total Cholesterol, mg/dL",
    "Baseline Total Cholesterol, mmol/L",
    "Baseline Triglycerides, mg/dL",
    "Baseline Triglycerides, mmol/L",
    "Baseline HbA1c, %",
    "Baseline HbA1c, mmol/mol",
    "Baseline Alcohol consumption, non:1, light:2, moderate:3, heavy:4",
    "Baseline Smoking status, 1:never, 2:past, 3:current",
    "Baseline Fasting plasma glucose, mg/dl",
    "Baseline Fasting plasma glucose, mmol/L",
    "Baseline Systolic blood pressure, mmHg",
    "Baseline Diastolic blood pressure, mmHg",
    "Incident DM 0/1",
)

RAW_TO_PROCESSED: Mapping[str, str | None] = {
    "Baseline Age, yrs": "age",
    "Sex, 1:Men, 2:Women": "gender",
    "Baseline BMI, kg/m2": "bmi",
    "Baseline Systolic blood pressure, mmHg": "sbp",
    "Baseline Diastolic blood pressure, mmHg": "dbp",
    "Baseline Fasting plasma glucose, mmol/L": "fpg",
    "Baseline Total Cholesterol, mmol/L": "total_cholesterol",
    "Baseline Triglycerides, mmol/L": "triglycerides",
    "Baseline ALT, IU/L": "alt",
    "Patient Number": "participant_id",
    "Follow up duration, days": "duration_years",
    "Incident DM 0/1": "event",
}

COMPATIBLE_BASE_FEATURES: tuple[str, ...] = (
    "age",
    "gender",
    "bmi",
    "sbp",
    "dbp",
)
UNAVAILABLE_BASE_FEATURES: tuple[str, ...] = ("family_history_diabetes",)
COMPATIBLE_ACQUIRABLE_FEATURES: tuple[str, ...] = (
    "fpg",
    "total_cholesterol",
    "triglycerides",
    "alt",
)
UNAVAILABLE_ACQUIRABLE_FEATURES: tuple[str, ...] = ("ccr", "bun")

CLINICAL_AUDIT_RANGES: Mapping[str, tuple[float, float, str]] = {
    "age": (18.0, 110.0, "years; broad adult sanity range"),
    "bmi": (10.0, 80.0, "kg/m2; broad adult sanity range"),
    "sbp": (50.0, 260.0, "mmHg; broad systolic sanity range"),
    "dbp": (30.0, 180.0, "mmHg; broad diastolic sanity range"),
    "fpg": (2.0, 7.0, "mmol/L; baseline diabetes-free plausibility range"),
    "total_cholesterol": (1.0, 20.0, "mmol/L; broad sanity range"),
    "triglycerides": (0.1, 50.0, "mmol/L; broad sanity range"),
    "alt": (1.0, 2000.0, "IU/L; broad sanity range"),
}


@dataclass(frozen=True)
class NagalaCohort:
    mapped: pd.DataFrame
    audit: Mapping[str, Any]


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_summary(series: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(series, errors="raise")
    quantiles = numeric.quantile([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    return {
        "count": int(numeric.notna().sum()),
        "minimum": float(quantiles.loc[0.0]),
        "p01": float(quantiles.loc[0.01]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p99": float(quantiles.loc[0.99]),
        "maximum": float(quantiles.loc[1.0]),
    }


def _column_group(column: str) -> str:
    lowered = column.lower()
    if column == "Patient Number":
        return "identifier"
    if column in {"Sex, 1:Men, 2:Women", "Baseline Age, yrs"}:
        return "demographics"
    if column in {
        "Baseline BMI, kg/m2",
        "Baseline Waist circumference(WC), cm",
        "Baseline BMI≥25 0/1",
        "Baseline WC≥90 in Men, ≥80 in Women 0/1",
        "Baseline Body Weight, kg",
        "Obesity phenotype 1-8",
    }:
        return "anthropometry/obesity"
    if column in {
        "Baseline Ethanol consumption g/wk",
        "Baseline Habit of exercise 0/1",
        "Baseline Alcohol consumption, non:1, light:2, moderate:3, heavy:4",
        "Baseline Smoking status, 1:never, 2:past, 3:current",
    }:
        return "lifestyle"
    if column == "Follow up duration, days":
        return "follow-up"
    if column == "Incident DM 0/1":
        return "diabetes outcome"
    if "blood pressure" in lowered:
        return "blood pressure"
    if any(token in lowered for token in ("alt", "ast", "ggt")):
        return "liver laboratory"
    if any(token in lowered for token in ("cholesterol", "triglycerides")):
        return "lipids"
    if any(token in lowered for token in ("glucose", "hba1c")):
        return "glucose"
    if column == "Baseline Fatty liver 0/1":
        return "fatty liver"
    return "other"


def _validate_raw_frame(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != RAW_COLUMNS:
        missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(RAW_COLUMNS))
        raise ValueError(
            f"NAGALA columns differ from the verified schema; missing={missing}, "
            f"unexpected={unexpected}"
        )
    if frame.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(f"Expected NAGALA shape {(EXPECTED_ROWS, EXPECTED_COLUMNS)}; found {frame.shape}.")
    identifier = frame["Patient Number"]
    if identifier.isna().any() or identifier.duplicated().any():
        raise ValueError("NAGALA Patient Number must be non-missing and unique.")
    event = frame["Incident DM 0/1"]
    if not event.isin((0, 1)).all() or int(event.sum()) != EXPECTED_EVENTS:
        raise ValueError("NAGALA incident-DM status differs from the published cohort.")
    duration = frame["Follow up duration, days"]
    if duration.isna().any() or (duration <= 0).any():
        raise ValueError("NAGALA follow-up duration must be non-missing and positive.")
    sex = frame["Sex, 1:Men, 2:Women"]
    if set(sex.unique()) != {0, 1}:
        raise ValueError("NAGALA observed sex coding must be the verified 0/1 values.")
    if int((sex == 1).sum()) != 8_430 or int((sex == 0).sum()) != 7_034:
        raise ValueError("NAGALA sex-code counts differ from the published cohort counts.")


def _mapped_frame(raw: pd.DataFrame) -> pd.DataFrame:
    mapped = pd.DataFrame(index=raw.index)
    mapped["participant_id"] = raw["Patient Number"].astype(np.int64)
    mapped["age"] = raw["Baseline Age, yrs"].astype(float)
    mapped["gender"] = raw["Sex, 1:Men, 2:Women"].map({1: 1, 0: 2}).astype(float)
    mapped["bmi"] = raw["Baseline BMI, kg/m2"].astype(float)
    mapped["sbp"] = raw["Baseline Systolic blood pressure, mmHg"].astype(float)
    mapped["dbp"] = raw["Baseline Diastolic blood pressure, mmHg"].astype(float)
    mapped["family_history_diabetes"] = np.nan
    mapped["fpg"] = raw["Baseline Fasting plasma glucose, mmol/L"].astype(float)
    mapped["total_cholesterol"] = raw["Baseline Total Cholesterol, mmol/L"].astype(float)
    mapped["triglycerides"] = raw["Baseline Triglycerides, mmol/L"].astype(float)
    mapped["alt"] = raw["Baseline ALT, IU/L"].astype(float)
    mapped["ccr"] = np.nan
    mapped["bun"] = np.nan
    mapped["event"] = raw["Incident DM 0/1"].astype(np.int64)
    mapped["duration_years"] = (
        raw["Follow up duration, days"].astype(float) / DAYS_PER_YEAR
    )
    if tuple(mapped.loc[:, PREDICTOR_FEATURES].columns) != PREDICTOR_FEATURES:
        raise ValueError("Mapped NAGALA predictors differ from the primary predictor order.")
    if mapped.loc[:, (*COMPATIBLE_BASE_FEATURES, *COMPATIBLE_ACQUIRABLE_FEATURES)].isna().any().any():
        raise ValueError("A compatible NAGALA predictor unexpectedly contains missing values.")
    return mapped


def _audit_document(path: Path, raw: pd.DataFrame, mapped: pd.DataFrame) -> dict[str, Any]:
    column_profiles = []
    for position, column in enumerate(raw.columns, start=1):
        series = raw[column]
        profile: dict[str, Any] = {
            "position": position,
            "name": column,
            "group": _column_group(column),
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "missing_percent": float(100.0 * series.isna().mean()),
            "unique_non_missing": int(series.nunique(dropna=True)),
            "mapped_to": RAW_TO_PROCESSED.get(column),
        }
        if (
            column != "Patient Number"
            and pd.api.types.is_numeric_dtype(series)
            and series.notna().any()
        ):
            profile["summary"] = _numeric_summary(series)
        column_profiles.append(profile)

    clinical_audit = []
    for feature, (lower, upper, rationale) in CLINICAL_AUDIT_RANGES.items():
        series = mapped[feature]
        clinical_audit.append(
            {
                "feature": feature,
                **_numeric_summary(series),
                "audit_lower": lower,
                "audit_upper": upper,
                "audit_rationale": rationale,
                "below": int((series < lower).sum()),
                "above": int((series > upper).sum()),
                "action": "retain; document only",
            }
        )

    duration = mapped["duration_years"]
    event = mapped["event"]
    return {
        "dataset_doi": DATASET_DOI,
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "md5": file_digest(path, "md5"),
        "sha256": file_digest(path, "sha256"),
        "workbook": {
            "sheets": [EXPECTED_SHEET],
            "data_rows": int(len(raw)),
            "columns": int(len(raw.columns)),
            "row_meaning": "one participant-level follow-up summary per row",
        },
        "integrity": {
            "duplicate_rows": int(raw.duplicated().sum()),
            "missing_identifiers": int(raw["Patient Number"].isna().sum()),
            "duplicate_identifiers": int(raw["Patient Number"].duplicated().sum()),
            "rows_excluded": 0,
        },
        "column_profiles": column_profiles,
        "outcome": {
            "event_raw_column": "Incident DM 0/1",
            "event_mapping": "1 = incident diabetes; 0 = right-censored",
            "events": int(event.sum()),
            "event_percent": float(100.0 * event.mean()),
            "duration_raw_column": "Follow up duration, days",
            "duration_mapping": f"duration_years = days / {DAYS_PER_YEAR}",
            "duration_years": _numeric_summary(duration),
            "censoring": "event = 0 at the recorded follow-up duration",
        },
        "sex_coding": {
            "header_claim": "1:Men, 2:Women",
            "observed_codes": {"0": 7_034, "1": 8_430},
            "published_counts": {"women": 7_034, "men": 8_430},
            "confirmed_mapping_to_primary": "raw 1 -> primary 1 (male); raw 0 -> primary 2 (female)",
        },
        "confirmed_schema_issues": {
            "sex_header_mismatch": (
                "header claims 1/2 but observed values are 0/1; published sex counts "
                "confirm 1 = male and 0 = female"
            ),
            "obesity_phenotype_header_mismatch": {
                "header_claim": "1-8",
                "observed_codes": {
                    str(key): int(value)
                    for key, value in raw["Obesity phenotype 1-8"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "handling": "not mapped or used by the primary predictors or policies",
            },
            "hdl_dual_unit_missingness": {
                "mg_dl_missing": int(raw["Baseline HDL-cholesterol, mg/dL"].isna().sum()),
                "mmol_l_zero": int((raw["Baseline HDL-cholesterol, mmol/L"] == 0).sum()),
                "handling": "HDL is optional and is not used in external validation",
            },
        },
        "feature_compatibility": {
            "compatible_base": list(COMPATIBLE_BASE_FEATURES),
            "unavailable_base": list(UNAVAILABLE_BASE_FEATURES),
            "compatible_acquirable": list(COMPATIBLE_ACQUIRABLE_FEATURES),
            "unavailable_acquirable": list(UNAVAILABLE_ACQUIRABLE_FEATURES),
            "unavailable_handling": (
                "family history is represented as source-missing and receives the primary "
                "train-fitted imputation; CCR and BUN are removed from external actions"
            ),
        },
        "clinical_audit": clinical_audit,
        "unit_consistency": {
            "fpg_max_abs_mmol_minus_mgdl_over_18": float(
                np.max(
                    np.abs(
                        raw["Baseline Fasting plasma glucose, mmol/L"]
                        - raw["Baseline Fasting plasma glucose, mg/dl"] / 18.0
                    )
                )
            ),
            "total_cholesterol_max_abs_mmol_minus_mgdl_times_0_02586": float(
                np.max(
                    np.abs(
                        raw["Baseline Total Cholesterol, mmol/L"]
                        - raw["Baseline Total Cholesterol, mg/dL"] * 0.02586
                    )
                )
            ),
            "triglycerides_max_abs_mmol_minus_mgdl_times_0_01129": float(
                np.max(
                    np.abs(
                        raw["Baseline Triglycerides, mmol/L"]
                        - raw["Baseline Triglycerides, mg/dL"] * 0.01129
                    )
                )
            ),
        },
    }


def load_nagala_cohort(path: Path) -> NagalaCohort:

    if not path.exists():
        raise FileNotFoundError(f"NAGALA workbook not found: {path}")
    if path.name != EXPECTED_FILE_NAME:
        raise ValueError(f"Expected NAGALA filename {EXPECTED_FILE_NAME!r}.")
    if path.stat().st_size != EXPECTED_SIZE_BYTES:
        raise ValueError("NAGALA file size differs from Dryad metadata.")
    if file_digest(path, "md5") != EXPECTED_MD5:
        raise ValueError("NAGALA MD5 differs from Dryad metadata.")
    workbook = pd.ExcelFile(path)
    if workbook.sheet_names != [EXPECTED_SHEET]:
        raise ValueError(f"Expected one NAGALA sheet {EXPECTED_SHEET!r}.")
    raw = pd.read_excel(workbook, sheet_name=EXPECTED_SHEET)
    _validate_raw_frame(raw)
    if set(COMPATIBLE_BASE_FEATURES) | set(UNAVAILABLE_BASE_FEATURES) != set(
        BASE_FEATURES
    ):
        raise RuntimeError("NAGALA base compatibility does not cover the primary base set.")
    if set(COMPATIBLE_ACQUIRABLE_FEATURES) | set(
        UNAVAILABLE_ACQUIRABLE_FEATURES
    ) != set(CORE_ACQUIRABLE_FEATURES):
        raise RuntimeError(
            "NAGALA acquisition compatibility does not cover the primary action set."
        )
    mapped = _mapped_frame(raw)
    return NagalaCohort(mapped=mapped, audit=_audit_document(path, raw, mapped))
