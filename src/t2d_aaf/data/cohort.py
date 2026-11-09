from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

from openpyxl import load_workbook


RAW_SHEET_NAME = "RC"
DEFAULT_SEED = 20260821

RAW_ID_COLUMN = "id"
RAW_EVENT_COLUMN = "censor of diabetes at followup(1, Yes; 0, No)"
RAW_DURATION_COLUMN = "year of followup"
RAW_FINAL_FPG_COLUMN = "FPG of final visit(mmol/L)"
RAW_DIAGNOSIS_FLAG_COLUMN = "Diabetes diagnosed during followup（1,Yes）"

RAW_COLUMN_TO_CLEAN_NAME: dict[str, str] = {
    RAW_ID_COLUMN: "participant_id",
    "Age (y)": "age",
    "Gender(1, male; 2, female)": "gender",
    "site": "site",
    "height(cm)": "height_cm",
    "weight(kg)": "weight_kg",
    "BMI(kg/m2)": "bmi",
    "SBP(mmHg)": "sbp",
    "DBP(mmHg)": "dbp",
    "FPG (mmol/L)": "fpg",
    "Cholesterol(mmol/L)": "total_cholesterol",
    "Triglyceride(mmol/L)": "triglycerides",
    "HDL-c(mmol/L)": "hdl_c",
    "LDL(mmol/L)": "ldl",
    "ALT(U/L)": "alt",
    "AST(U/L)": "ast",
    "BUN(mmol/L)": "bun",
    "CCR(umol/L)": "ccr",
    RAW_EVENT_COLUMN: "event",
    RAW_DURATION_COLUMN: "duration_years",
    "smoking status(1,current smoker;2, ever smoker;3,never smoker)": "smoking_status",
    "drinking status(1,current drinker;2, ever drinker;3,never drinker)": "drinking_status",
    "family histroy of diabetes(1,Yes;0,No)": "family_history_diabetes",
}

TARGET_COLUMNS: tuple[str, ...] = ("event", "duration_years")
METADATA_COLUMNS: tuple[str, ...] = ("participant_id", "site", "split")

BASE_FEATURES: tuple[str, ...] = (
    "age",
    "gender",
    "bmi",
    "sbp",
    "dbp",
    "family_history_diabetes",
)

CORE_ACQUIRABLE_FEATURES: tuple[str, ...] = (
    "fpg",
    "total_cholesterol",
    "triglycerides",
    "alt",
    "ccr",
    "bun",
)

OPTIONAL_FEATURES: tuple[str, ...] = (
    "height_cm",
    "weight_kg",
    "smoking_status",
    "drinking_status",
    "hdl_c",
    "ldl",
    "ast",
)

EXCLUDED_RAW_FIELDS: tuple[str, ...] = (
    RAW_FINAL_FPG_COLUMN,
    RAW_DIAGNOSIS_FLAG_COLUMN,
    RAW_EVENT_COLUMN,
    RAW_DURATION_COLUMN,
    RAW_ID_COLUMN,
)

EXCLUDED_FROM_MAIN_PREDICTORS: tuple[str, ...] = (
    "participant_id",
    "site",
    "event",
    "duration_years",
    "final_fpg",
    "diabetes_diagnosed_followup",
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "participant_id",
    "split",
    "event",
    "duration_years",
    "site",
    *BASE_FEATURES,
    *CORE_ACQUIRABLE_FEATURES,
    *OPTIONAL_FEATURES,
)

CONTINUOUS_AUDIT_FEATURES: tuple[str, ...] = (
    "age",
    "bmi",
    "sbp",
    "dbp",
    "fpg",
    "total_cholesterol",
    "triglycerides",
    "alt",
    "ccr",
    "bun",
)

CLINICAL_AUDIT_RANGES: dict[str, tuple[float, float, str]] = {
    "age": (18.0, 110.0, "adult age sanity range"),
    "bmi": (10.0, 80.0, "broad adult BMI sanity range"),
    "sbp": (50.0, 260.0, "broad systolic BP sanity range"),
    "dbp": (30.0, 180.0, "broad diastolic BP sanity range"),
    "fpg": (2.0, 7.0, "baseline diabetes-free cohort plausibility range"),
    "total_cholesterol": (1.0, 20.0, "broad total cholesterol sanity range"),
    "triglycerides": (0.1, 50.0, "broad triglyceride sanity range"),
    "alt": (1.0, 2000.0, "broad ALT sanity range"),
    "ccr": (10.0, 1500.0, "broad creatinine/CCR sanity range using file units"),
    "bun": (0.5, 50.0, "broad BUN sanity range"),
}

CLEANING_RULES: tuple[dict[str, str], ...] = (
    {
        "rule": "missing_id",
        "action": "exclude row",
        "reason": "participant-level split requires a non-missing participant identifier",
    },
    {
        "rule": "duplicate_id",
        "action": "exclude duplicate participant rows if present",
        "reason": "one participant must not appear in multiple splits",
    },
    {
        "rule": "invalid_event",
        "action": "exclude row",
        "reason": "event/status must be exactly 0 or 1",
    },
    {
        "rule": "invalid_duration",
        "action": "exclude row",
        "reason": "survival duration must be numeric and greater than zero",
    },
    {
        "rule": "missing_predictor_or_extreme_value",
        "action": "retain row and value",
        "reason": "no imputation, scaling, or arbitrary clinical outlier deletion is performed before split",
    },
)


@dataclass(frozen=True)
class CohortBuildResult:
    records: list[dict[str, Any]]
    raw_row_count: int
    row_exclusions: dict[str, int]
    duplicate_participant_ids: list[int]
    seed: int
    split_fractions: dict[str, float]
    parquet_status: dict[str, Any]


def find_primary_raw_path(project_root: Path) -> Path:
    candidates = (
        project_root / "data" / "raw" / "RC+Health+Care+Data-20180820.xlsx",
        project_root / "data" / "raw" / "RC Health Care Data-20180820.xlsx",
    )
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"Primary workbook not found. Checked:\n{checked}")


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def parse_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        if not text:
            return None
        return float(text)
    raise TypeError(f"Cannot parse numeric value from {value!r}")


def parse_int(value: Any) -> int | None:
    number = parse_float(value)
    if number is None:
        return None
    if not math.isfinite(number) or abs(number - round(number)) > 1e-9:
        raise ValueError(f"Expected integer-like value, got {value!r}")
    return int(round(number))


def clean_raw_record(raw_record: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:

    reasons: list[str] = []
    participant_id = parse_int(raw_record.get(RAW_ID_COLUMN))
    event = parse_int(raw_record.get(RAW_EVENT_COLUMN))
    duration = parse_float(raw_record.get(RAW_DURATION_COLUMN))

    if participant_id is None:
        reasons.append("missing_id")
    if event not in (0, 1):
        reasons.append("invalid_event")
    if duration is None or not math.isfinite(duration) or duration <= 0:
        reasons.append("invalid_duration")
    if reasons:
        return None, reasons

    record: dict[str, Any] = {
        "participant_id": participant_id,
        "event": event,
        "duration_years": duration,
        "site": parse_int(raw_record.get("site")),
        "age": parse_int(raw_record.get("Age (y)")),
        "gender": parse_int(raw_record.get("Gender(1, male; 2, female)")),
        "bmi": parse_float(raw_record.get("BMI(kg/m2)")),
        "sbp": parse_float(raw_record.get("SBP(mmHg)")),
        "dbp": parse_float(raw_record.get("DBP(mmHg)")),
        "family_history_diabetes": parse_int(raw_record.get("family histroy of diabetes(1,Yes;0,No)")),
        "fpg": parse_float(raw_record.get("FPG (mmol/L)")),
        "total_cholesterol": parse_float(raw_record.get("Cholesterol(mmol/L)")),
        "triglycerides": parse_float(raw_record.get("Triglyceride(mmol/L)")),
        "alt": parse_float(raw_record.get("ALT(U/L)")),
        "ccr": parse_float(raw_record.get("CCR(umol/L)")),
        "bun": parse_float(raw_record.get("BUN(mmol/L)")),
        "height_cm": parse_float(raw_record.get("height(cm)")),
        "weight_kg": parse_float(raw_record.get("weight(kg)")),
        "smoking_status": parse_int(raw_record.get("smoking status(1,current smoker;2, ever smoker;3,never smoker)")),
        "drinking_status": parse_int(raw_record.get("drinking status(1,current drinker;2, ever drinker;3,never drinker)")),
        "hdl_c": parse_float(raw_record.get("HDL-c(mmol/L)")),
        "ldl": parse_float(raw_record.get("LDL(mmol/L)")),
        "ast": parse_float(raw_record.get("AST(U/L)")),
    }
    return record, []


def load_clean_records(raw_path: Path) -> tuple[list[dict[str, Any]], int, dict[str, int], list[int]]:
    workbook = load_workbook(raw_path, read_only=True, data_only=True)
    if RAW_SHEET_NAME not in workbook.sheetnames:
        raise ValueError(f"Expected sheet {RAW_SHEET_NAME!r}; found {workbook.sheetnames!r}")
    sheet = workbook[RAW_SHEET_NAME]
    header = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
    raw_columns = [str(value).strip() for value in header]
    missing_columns = [column for column in RAW_COLUMN_TO_CLEAN_NAME if column not in raw_columns]
    for column in (RAW_FINAL_FPG_COLUMN, RAW_DIAGNOSIS_FLAG_COLUMN):
        if column not in raw_columns:
            missing_columns.append(column)
    if missing_columns:
        raise ValueError("Missing expected raw columns: " + ", ".join(missing_columns))

    index_by_column = {column: index for index, column in enumerate(raw_columns)}
    records: list[dict[str, Any]] = []
    row_exclusions: Counter[str] = Counter()
    raw_row_count = 0

    needed_columns = set(RAW_COLUMN_TO_CLEAN_NAME) | {RAW_FINAL_FPG_COLUMN, RAW_DIAGNOSIS_FLAG_COLUMN}
    for row in sheet.iter_rows(min_row=2, values_only=True):
        raw_row_count += 1
        raw_record = {column: row[index_by_column[column]] for column in needed_columns}
        record, reasons = clean_raw_record(raw_record)
        if reasons:
            for reason in reasons:
                row_exclusions[reason] += 1
            continue
        if record is not None:
            records.append(record)

    id_counts = Counter(record["participant_id"] for record in records)
    duplicate_ids = sorted(int(identifier) for identifier, count in id_counts.items() if count > 1)
    if duplicate_ids:
        duplicate_set = set(duplicate_ids)
        row_exclusions["duplicate_id"] += sum(
            1 for record in records if record["participant_id"] in duplicate_set
        )
        records = [record for record in records if record["participant_id"] not in duplicate_set]

    for rule in ("missing_id", "duplicate_id", "invalid_event", "invalid_duration"):
        row_exclusions.setdefault(rule, 0)

    return records, raw_row_count, dict(row_exclusions), duplicate_ids


def split_participant_ids(
    records: list[dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[int, str]:
    if not records:
        raise ValueError("Cannot split an empty cohort.")
    test_fraction = 1.0 - train_fraction - validation_fraction
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("Split fractions must be positive and sum to less than or equal to 1.")

    rng = random.Random(seed)
    split_by_id: dict[int, str] = {}
    for event_value in (0, 1):
        ids = sorted(int(record["participant_id"]) for record in records if record["event"] == event_value)
        if not ids:
            raise ValueError(f"No records found for event={event_value}.")
        rng.shuffle(ids)
        n_total = len(ids)
        n_train = round(n_total * train_fraction)
        n_validation = round(n_total * validation_fraction)
        train_ids = ids[:n_train]
        validation_ids = ids[n_train : n_train + n_validation]
        test_ids = ids[n_train + n_validation :]
        for participant_id in train_ids:
            split_by_id[participant_id] = "train"
        for participant_id in validation_ids:
            split_by_id[participant_id] = "validation"
        for participant_id in test_ids:
            split_by_id[participant_id] = "test"

    if len(split_by_id) != len(records):
        raise ValueError("Split assignment count does not match cohort size.")
    return split_by_id


def apply_split(records: list[dict[str, Any]], split_by_id: dict[int, str]) -> list[dict[str, Any]]:
    cohort = []
    for record in records:
        participant_id = int(record["participant_id"])
        row = {column: record.get(column) for column in OUTPUT_COLUMNS if column != "split"}
        row["split"] = split_by_id[participant_id]
        cohort.append(row)
    return sorted(cohort, key=lambda row: int(row["participant_id"]))


def build_primary_cohort(raw_path: Path, *, seed: int = DEFAULT_SEED) -> CohortBuildResult:
    records, raw_row_count, row_exclusions, duplicate_ids = load_clean_records(raw_path)
    split_by_id = split_participant_ids(records, seed=seed)
    cohort_records = apply_split(records, split_by_id)
    return CohortBuildResult(
        records=cohort_records,
        raw_row_count=raw_row_count,
        row_exclusions=row_exclusions,
        duplicate_participant_ids=duplicate_ids,
        seed=seed,
        split_fractions={"train": 0.70, "validation": 0.15, "test": 0.15},
        parquet_status={"written": False, "reason": "not attempted"},
    )


def quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    clean_values = [value for value in values if value is not None and math.isfinite(value)]
    if not clean_values:
        return {
            "count": 0,
            "min": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
            "sd": None,
        }
    sorted_values = sorted(clean_values)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "p01": quantile(sorted_values, 0.01),
        "p05": quantile(sorted_values, 0.05),
        "p25": quantile(sorted_values, 0.25),
        "median": quantile(sorted_values, 0.50),
        "p75": quantile(sorted_values, 0.75),
        "p95": quantile(sorted_values, 0.95),
        "p99": quantile(sorted_values, 0.99),
        "max": sorted_values[-1],
        "mean": statistics.fmean(sorted_values),
        "sd": statistics.stdev(sorted_values) if len(sorted_values) > 1 else 0.0,
    }


def values_for(records: list[dict[str, Any]], column: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(column)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                values.append(float(value))
    return values


def missing_count(records: list[dict[str, Any]], column: str) -> int:
    return sum(1 for record in records if is_missing(record.get(column)))


def split_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        split_records = [record for record in records if record["split"] == split]
        event_count = sum(int(record["event"]) for record in split_records)
        summary[split] = {
            "participants": len(split_records),
            "events": event_count,
            "event_percent": 100.0 * event_count / len(split_records),
            "duration": numeric_summary(values_for(split_records, "duration_years")),
        }
    return summary


def clinical_audit(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    for feature in CONTINUOUS_AUDIT_FEATURES:
        values = values_for(records, feature)
        low, high, note = CLINICAL_AUDIT_RANGES[feature]
        audit[feature] = {
            **numeric_summary(values),
            "missing": missing_count(records, feature),
            "range_low": low,
            "range_high": high,
            "below_range": sum(1 for value in values if value < low),
            "above_range": sum(1 for value in values if value > high),
            "note": note,
            "cleaning_action": "retained; documented only",
        }
    return audit


def predictor_missingness_by_split(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    features = (*BASE_FEATURES, *CORE_ACQUIRABLE_FEATURES, *OPTIONAL_FEATURES)
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for feature in features:
        result[feature] = {}
        for split in ("train", "validation", "test"):
            split_records = [record for record in records if record["split"] == split]
            missing = missing_count(split_records, feature)
            result[feature][split] = {
                "missing": missing,
                "missing_percent": 100.0 * missing / len(split_records),
            }
    return result


def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({column: "" if record.get(column) is None else record.get(column) for column in OUTPUT_COLUMNS})


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def try_write_parquet(records: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr_buffer):
            import pyarrow as pa
            import pyarrow.parquet as pq
    except Exception as exc:
        return {
            "written": False,
            "path": portable_path(output_path),
            "reason": f"{type(exc).__name__}: {exc}",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([{column: record.get(column) for column in OUTPUT_COLUMNS} for record in records])
    pq.write_table(table, output_path)
    return {"written": True, "path": portable_path(output_path), "reason": ""}


def write_manifest(
    result: CohortBuildResult,
    *,
    raw_path: Path,
    csv_path: Path,
    parquet_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "primary_cohort_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_workbook": portable_path(raw_path),
        "source_sheet": RAW_SHEET_NAME,
        "raw_row_count": result.raw_row_count,
        "cohort_row_count": len(result.records),
        "seed": result.seed,
        "split_fractions": result.split_fractions,
        "target_columns": list(TARGET_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS),
        "base_features": list(BASE_FEATURES),
        "core_acquirable_features": list(CORE_ACQUIRABLE_FEATURES),
        "optional_features": list(OPTIONAL_FEATURES),
        "excluded_raw_fields": list(EXCLUDED_RAW_FIELDS),
        "excluded_from_main_predictors": list(EXCLUDED_FROM_MAIN_PREDICTORS),
        "cleaning_rules": list(CLEANING_RULES),
        "row_exclusions": result.row_exclusions,
        "duplicate_participant_ids": result.duplicate_participant_ids[:20],
        "outputs": {
            "csv": portable_path(csv_path),
            "parquet": portable_path(parquet_path) if result.parquet_status.get("written") else None,
            "parquet_status": result.parquet_status,
            "manifest": portable_path(manifest_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
        return text
    return str(value)


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def feature_role(feature: str) -> str:
    if feature in BASE_FEATURES:
        return "base"
    if feature in CORE_ACQUIRABLE_FEATURES:
        return "core acquirable"
    if feature in OPTIONAL_FEATURES:
        return "optional"
    return "other"


def build_cohort_report(
    result: CohortBuildResult,
    *,
    raw_path: Path,
    csv_path: Path,
    parquet_path: Path,
    manifest_path: Path,
) -> str:
    records = result.records
    split_stats = split_summary(records)
    clinical = clinical_audit(records)
    missingness = predictor_missingness_by_split(records)
    total_events = sum(int(record["event"]) for record in records)
    total_participants = len(records)

    lines: list[str] = [
        "# Primary Cohort Summary",
        "",
        "cohort construction cohort construction for the primary Rui-Ci / Rich Healthcare Group dataset.",
        "",
        "No model training, imputation fitting, scaling fitting, or Active Feature Acquisition policy is performed in this stage.",
        "",
        "## Source And Output",
        "",
        f"- Source workbook: `{portable_path(raw_path)}`",
        f"- Source sheet: `{RAW_SHEET_NAME}`",
        f"- Raw rows read: {result.raw_row_count:,}",
        f"- Cohort rows retained: {total_participants:,}",
        f"- Event target: `event = {RAW_EVENT_COLUMN}`",
        f"- Duration target: `duration_years = {RAW_DURATION_COLUMN}`",
        f"- Split seed: `{result.seed}`",
        f"- CSV cohort: `{portable_path(csv_path)}`",
        f"- Manifest: `{portable_path(manifest_path)}`",
    ]
    if result.parquet_status.get("written"):
        lines.append(f"- Parquet cohort: `{portable_path(parquet_path)}`")
    else:
        lines.append(
            "- Parquet cohort: not written in this environment; "
            f"{result.parquet_status.get('reason', 'optional engine unavailable')}. CSV is the canonical local output."
        )

    lines.extend(
        [
            "",
            "## Cleaning Rules",
            "",
            "| Rule | Action | Reason | Rows affected |",
            "|---|---|---|---:|",
        ]
    )
    for rule in CLEANING_RULES:
        rule_name = rule["rule"]
        affected = result.row_exclusions.get(rule_name, 0)
        if rule_name == "missing_predictor_or_extreme_value":
            affected = 0
        lines.append(
            f"| {md_escape(rule_name)} | {md_escape(rule['action'])} | {md_escape(rule['reason'])} | {affected:,} |"
        )

    lines.extend(
        [
            "",
            "Continuous predictor missingness and suspicious values were audited, but no predictor values were imputed, scaled, winsorized, or deleted before the split.",
            "",
            "## Feature Sets",
            "",
            "- Base features: " + ", ".join(f"`{feature}`" for feature in BASE_FEATURES),
            "- Core acquirable laboratory features: "
            + ", ".join(f"`{feature}`" for feature in CORE_ACQUIRABLE_FEATURES),
            "- Optional features retained for later sensitivity work: "
            + ", ".join(f"`{feature}`" for feature in OPTIONAL_FEATURES),
            "- Metadata retained outside predictors: `participant_id`, `site`, `split`",
            "- Excluded/leakage fields: "
            + ", ".join(f"`{field}`" for field in EXCLUDED_RAW_FIELDS),
            "",
            "`site` is retained only as metadata for future sensitivity analysis and is not part of the main predictor set.",
            "`height_cm` and `weight_kg` are optional only; the main base set uses `bmi` instead of including height and weight simultaneously.",
            "`smoking_status` and `drinking_status` are optional only because data inspection found about 71.6% missingness.",
            "`hdl_c`, `ldl`, and `ast` are optional only because data inspection found high missingness.",
            "",
            "## Split Summary",
            "",
            "| Split | Participants | Events | Event % |",
            "|---|---:|---:|---:|",
        ]
    )
    for split in ("train", "validation", "test"):
        stats = split_stats[split]
        lines.append(
            f"| {split} | {stats['participants']:,} | {stats['events']:,} | {stats['event_percent']:.2f} |"
        )
    lines.append(f"| total | {total_participants:,} | {total_events:,} | {100.0 * total_events / total_participants:.2f} |")

    lines.extend(
        [
            "",
            "## Duration Distribution By Split",
            "",
            "| Split | Count | Min | P25 | Median | P75 | Max | Mean | SD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "validation", "test"):
        duration = split_stats[split]["duration"]
        lines.append(
            f"| {split} | {duration['count']:,} | {format_number(duration['min'])} | "
            f"{format_number(duration['p25'])} | {format_number(duration['median'])} | "
            f"{format_number(duration['p75'])} | {format_number(duration['max'])} | "
            f"{format_number(duration['mean'])} | {format_number(duration['sd'])} |"
        )

    lines.extend(
        [
            "",
            "## Predictor Missingness By Split",
            "",
            "| Feature | Role | Train missing | Train % | Validation missing | Validation % | Test missing | Test % |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for feature in (*BASE_FEATURES, *CORE_ACQUIRABLE_FEATURES, *OPTIONAL_FEATURES):
        train = missingness[feature]["train"]
        validation = missingness[feature]["validation"]
        test = missingness[feature]["test"]
        lines.append(
            f"| {feature} | {feature_role(feature)} | "
            f"{train['missing']:,} | {train['missing_percent']:.2f} | "
            f"{validation['missing']:,} | {validation['missing_percent']:.2f} | "
            f"{test['missing']:,} | {test['missing_percent']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Continuous Feature Audit",
            "",
            "| Feature | Count | Missing | Min | P01 | P05 | P25 | Median | P75 | P95 | P99 | Max | Audit range | Below | Above | Cleaning action |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for feature in CONTINUOUS_AUDIT_FEATURES:
        audit = clinical[feature]
        lines.append(
            f"| {feature} | {audit['count']:,} | {audit['missing']:,} | "
            f"{format_number(audit['min'])} | {format_number(audit['p01'])} | {format_number(audit['p05'])} | "
            f"{format_number(audit['p25'])} | {format_number(audit['median'])} | {format_number(audit['p75'])} | "
            f"{format_number(audit['p95'])} | {format_number(audit['p99'])} | {format_number(audit['max'])} | "
            f"{format_number(audit['range_low'])}-{format_number(audit['range_high'])}; {md_escape(audit['note'])} | "
            f"{audit['below_range']:,} | {audit['above_range']:,} | {md_escape(audit['cleaning_action'])} |"
        )

    lines.extend(
        [
            "",
            "## Leakage Check",
            "",
            "- Main predictor features contain no `participant_id`, `site`, final-visit FPG, diagnosis flag, event/status column, or follow-up duration.",
            "- `participant_id` is retained only for split integrity and traceability.",
            "- `site` is retained only as metadata for future sensitivity analysis.",
            "- `event` and `duration_years` are targets, not predictors.",
            "- `FPG of final visit(mmol/L)` and `Diabetes diagnosed during followup（1,Yes）` are not written to the processed cohort.",
            "",
            "## Split Integrity",
            "",
            "The raw audit found one row per participant and unique participant IDs. The cohort construction split assigns each `participant_id` to exactly one split after stratifying by `event`.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_report(report_text: str, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
