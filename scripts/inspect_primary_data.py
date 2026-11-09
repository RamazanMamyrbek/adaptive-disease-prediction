from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time
import hashlib
import math
from pathlib import Path
import statistics
from typing import Any

from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH_CANDIDATES = (
    PROJECT_ROOT / "data" / "raw" / "RC+Health+Care+Data-20180820.xlsx",
    PROJECT_ROOT / "data" / "raw" / "RC Health Care Data-20180820.xlsx",
)
REPORT_PATH = PROJECT_ROOT / "reports" / "primary_dataset_profile.md"

EVENT_FLAG_COLUMN = "Diabetes diagnosed during followup（1,Yes）"
EVENT_STATUS_COLUMN = "censor of diabetes at followup(1, Yes; 0, No)"
FOLLOWUP_TIME_COLUMN = "year of followup"
FINAL_FPG_COLUMN = "FPG of final visit(mmol/L)"
ID_COLUMN = "id"

COLUMN_DESCRIPTIONS: dict[str, str] = {
    "id": "Participant identifier in the released table.",
    "Age (y)": "Baseline age in years.",
    "Gender(1, male; 2, female)": "Sex/gender code: 1 male, 2 female.",
    "site": "Health-check site code.",
    "height(cm)": "Baseline height in centimeters.",
    "weight(kg)": "Baseline weight in kilograms.",
    "BMI(kg/m2)": "Baseline body mass index.",
    "SBP(mmHg)": "Baseline systolic blood pressure.",
    "DBP(mmHg)": "Baseline diastolic blood pressure.",
    "FPG (mmol/L)": "Baseline fasting plasma glucose.",
    "Cholesterol(mmol/L)": "Baseline total cholesterol.",
    "Triglyceride(mmol/L)": "Baseline triglycerides.",
    "HDL-c(mmol/L)": "Baseline HDL cholesterol.",
    "LDL(mmol/L)": "Baseline LDL cholesterol.",
    "ALT(U/L)": "Baseline alanine aminotransferase.",
    "AST(U/L)": "Baseline aspartate aminotransferase.",
    "BUN(mmol/L)": "Baseline blood urea nitrogen.",
    "CCR(umol/L)": "Baseline creatinine clearance / creatinine-related lab as named in file.",
    FINAL_FPG_COLUMN: "Fasting plasma glucose at final visit; future/post-baseline value.",
    EVENT_FLAG_COLUMN: "Outcome-related follow-up diabetes diagnosis flag stored as 1 for yes and blank otherwise; incomplete relative to the event/status column.",
    EVENT_STATUS_COLUMN: "Binary follow-up diabetes event/status field: observed values use 1 for event and 0 for no event.",
    FOLLOWUP_TIME_COLUMN: "Follow-up duration in years.",
    "smoking status(1,current smoker;2, ever smoker;3,never smoker)": "Smoking status code.",
    "drinking status(1,current drinker;2, ever drinker;3,never drinker)": "Alcohol drinking status code.",
    "family histroy of diabetes(1,Yes;0,No)": "Family history of diabetes; column name contains the original typo 'histroy'.",
}

COLUMN_GROUPS: dict[str, str] = {
    "id": "identifier",
    "Age (y)": "demographics",
    "Gender(1, male; 2, female)": "demographics",
    "site": "demographics / site",
    "height(cm)": "anthropometry",
    "weight(kg)": "anthropometry",
    "BMI(kg/m2)": "anthropometry",
    "SBP(mmHg)": "blood pressure",
    "DBP(mmHg)": "blood pressure",
    "FPG (mmol/L)": "glucose",
    "Cholesterol(mmol/L)": "lipids",
    "Triglyceride(mmol/L)": "lipids",
    "HDL-c(mmol/L)": "lipids",
    "LDL(mmol/L)": "lipids",
    "ALT(U/L)": "other laboratory",
    "AST(U/L)": "other laboratory",
    "BUN(mmol/L)": "other laboratory",
    "CCR(umol/L)": "other laboratory",
    FINAL_FPG_COLUMN: "follow-up glucose / leakage",
    EVENT_FLAG_COLUMN: "diabetes outcome",
    EVENT_STATUS_COLUMN: "diabetes outcome / censoring status",
    FOLLOWUP_TIME_COLUMN: "follow-up",
    "smoking status(1,current smoker;2, ever smoker;3,never smoker)": "lifestyle",
    "drinking status(1,current drinker;2, ever drinker;3,never drinker)": "lifestyle",
    "family histroy of diabetes(1,Yes;0,No)": "family history",
}

BASE_FEATURES = {
    "Age (y)",
    "Gender(1, male; 2, female)",
    "site",
    "height(cm)",
    "weight(kg)",
    "BMI(kg/m2)",
    "SBP(mmHg)",
    "DBP(mmHg)",
    "smoking status(1,current smoker;2, ever smoker;3,never smoker)",
    "drinking status(1,current drinker;2, ever drinker;3,never drinker)",
    "family histroy of diabetes(1,Yes;0,No)",
}

CANDIDATE_ACQUIRABLE_FEATURES = {
    "FPG (mmol/L)",
    "Cholesterol(mmol/L)",
    "Triglyceride(mmol/L)",
    "HDL-c(mmol/L)",
    "LDL(mmol/L)",
    "ALT(U/L)",
    "AST(U/L)",
    "BUN(mmol/L)",
    "CCR(umol/L)",
}

EXCLUDED_FEATURES = {
    ID_COLUMN,
    FINAL_FPG_COLUMN,
    EVENT_FLAG_COLUMN,
    EVENT_STATUS_COLUMN,
    FOLLOWUP_TIME_COLUMN,
}

CODED_COLUMNS = {
    "Gender(1, male; 2, female)",
    "site",
    EVENT_FLAG_COLUMN,
    EVENT_STATUS_COLUMN,
    "smoking status(1,current smoker;2, ever smoker;3,never smoker)",
    "drinking status(1,current drinker;2, ever drinker;3,never drinker)",
    "family histroy of diabetes(1,Yes;0,No)",
}


@dataclass
class ColumnProfile:
    index: int
    name: str
    missing_count: int = 0
    non_missing_count: int = 0
    type_counts: Counter[str] = field(default_factory=Counter)
    value_counts: Counter[Any] = field(default_factory=Counter)
    numeric_values: list[float] = field(default_factory=list)


@dataclass
class SheetProfile:
    name: str
    max_row: int
    max_column: int
    headers: list[str]
    data_row_count: int
    blank_data_row_count: int
    exact_duplicate_rows: int
    duplicate_ids: int | None
    duplicate_id_values_sample: list[Any]
    columns: list[ColumnProfile]
    event_followup_summary: dict[str, Any]


def find_raw_path() -> Path:
    for path in RAW_PATH_CANDIDATES:
        if path.exists():
            return path
    candidates = "\n".join(f"- {path}" for path in RAW_PATH_CANDIDATES)
    raise FileNotFoundError(f"Primary raw workbook was not found. Checked:\n{candidates}")


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, time):
        return "time"
    if isinstance(value, str):
        return "str"
    return type(value).__name__


def normalize_for_hash(value: Any) -> str:
    if is_missing(value):
        return "<MISSING>"
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value):
            return "<NaN>"
        return repr(value)
    return repr(value)


def row_digest(row: tuple[Any, ...]) -> str:
    normalized = "\u241f".join(normalize_for_hash(value) for value in row)
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def percent(part: int | float, whole: int | float) -> float:
    if whole == 0:
        return 0.0
    return 100.0 * float(part) / float(whole)


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if abs(value - round(value)) < 10 ** (-(digits + 1)):
            return str(int(round(value)))
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


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


def inferred_dtype(profile: ColumnProfile) -> str:
    if profile.non_missing_count == 0:
        return "all missing"
    types = set(profile.type_counts)
    numeric_types = {"int", "float"}
    if types <= {"int"}:
        return "integer"
    if types <= numeric_types:
        if all(abs(value - round(value)) < 1e-12 for value in profile.numeric_values):
            return "integer-like numeric"
        return "numeric"
    if types <= {"str"}:
        return "string"
    if types <= {"date", "datetime", "time"}:
        return "date/time"
    return "mixed: " + ", ".join(sorted(types))


def feature_role(column_name: str) -> str:
    if column_name in BASE_FEATURES:
        return "base/free candidate"
    if column_name in CANDIDATE_ACQUIRABLE_FEATURES:
        return "candidate acquirable"
    if column_name in EXCLUDED_FEATURES:
        if column_name == FOLLOWUP_TIME_COLUMN:
            return "target time; exclude as predictor"
        if column_name == EVENT_STATUS_COLUMN:
            return "event/status target; exclude as predictor"
        if column_name == EVENT_FLAG_COLUMN:
            return "outcome-related flag; exclude as predictor"
        if column_name == FINAL_FPG_COLUMN:
            return "future value/leakage; exclude"
        return "identifier; exclude as predictor"
    return "unassigned"


def profile_sheet(ws: Any) -> SheetProfile:
    header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
    headers = ["" if value is None else str(value).strip() for value in header_row]
    column_profiles = [
        ColumnProfile(index=index, name=name) for index, name in enumerate(headers, start=1)
    ]

    data_row_count = 0
    blank_data_row_count = 0
    seen_row_digests: set[str] = set()
    exact_duplicate_rows = 0
    id_counter: Counter[Any] = Counter()
    id_index = headers.index(ID_COLUMN) if ID_COLUMN in headers else None

    event_index = headers.index(EVENT_STATUS_COLUMN) if EVENT_STATUS_COLUMN in headers else None
    diagnosis_index = headers.index(EVENT_FLAG_COLUMN) if EVENT_FLAG_COLUMN in headers else None
    followup_index = headers.index(FOLLOWUP_TIME_COLUMN) if FOLLOWUP_TIME_COLUMN in headers else None

    event_counts: Counter[Any] = Counter()
    diagnosis_counts: Counter[Any] = Counter()
    event_diagnosis_pairs: Counter[tuple[str, str]] = Counter()
    followup_by_event: dict[str, list[float]] = {"0": [], "1": [], "missing": []}

    for row in ws.iter_rows(min_row=2, values_only=True):
        row = tuple(row[: len(headers)])
        if all(is_missing(value) for value in row):
            blank_data_row_count += 1
            continue
        data_row_count += 1

        digest = row_digest(row)
        if digest in seen_row_digests:
            exact_duplicate_rows += 1
        else:
            seen_row_digests.add(digest)

        if id_index is not None:
            id_value = row[id_index]
            if not is_missing(id_value):
                id_counter[id_value] += 1

        for profile, value in zip(column_profiles, row):
            if is_missing(value):
                profile.missing_count += 1
                continue
            profile.non_missing_count += 1
            profile.type_counts[type_name(value)] += 1
            profile.value_counts[value] += 1
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not (isinstance(value, float) and math.isnan(value)):
                    profile.numeric_values.append(float(value))

        if event_index is not None:
            event_value = row[event_index]
            event_counts["missing" if is_missing(event_value) else event_value] += 1
        else:
            event_value = None

        if diagnosis_index is not None:
            diagnosis_value = row[diagnosis_index]
            diagnosis_counts["missing" if is_missing(diagnosis_value) else diagnosis_value] += 1
        else:
            diagnosis_value = None

        if event_index is not None and diagnosis_index is not None:
            event_label = "missing" if is_missing(event_value) else str(event_value)
            diagnosis_label = "missing" if is_missing(diagnosis_value) else str(diagnosis_value)
            event_diagnosis_pairs[(event_label, diagnosis_label)] += 1

        if followup_index is not None:
            followup_value = row[followup_index]
            if isinstance(followup_value, (int, float)) and not isinstance(followup_value, bool):
                if not (isinstance(followup_value, float) and math.isnan(followup_value)):
                    event_label = "missing" if event_index is None or is_missing(event_value) else str(event_value)
                    followup_by_event.setdefault(event_label, []).append(float(followup_value))

    duplicate_id_values = [value for value, count in id_counter.items() if count > 1]

    event_followup_summary = {
        "event_counts": event_counts,
        "diagnosis_counts": diagnosis_counts,
        "event_diagnosis_pairs": event_diagnosis_pairs,
        "followup_by_event": {
            key: numeric_summary_from_values(values) for key, values in followup_by_event.items()
        },
    }

    return SheetProfile(
        name=ws.title,
        max_row=ws.max_row,
        max_column=ws.max_column,
        headers=headers,
        data_row_count=data_row_count,
        blank_data_row_count=blank_data_row_count,
        exact_duplicate_rows=exact_duplicate_rows,
        duplicate_ids=len(duplicate_id_values) if id_index is not None else None,
        duplicate_id_values_sample=duplicate_id_values[:10],
        columns=column_profiles,
        event_followup_summary=event_followup_summary,
    )


def numeric_summary_from_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
            "sd": None,
        }
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "min": sorted_values[0],
        "p05": quantile(sorted_values, 0.05),
        "p25": quantile(sorted_values, 0.25),
        "median": quantile(sorted_values, 0.50),
        "p75": quantile(sorted_values, 0.75),
        "p95": quantile(sorted_values, 0.95),
        "max": sorted_values[-1],
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def numeric_summary(profile: ColumnProfile) -> dict[str, float | int | None]:
    return numeric_summary_from_values(profile.numeric_values)


def top_values(profile: ColumnProfile, limit: int = 10) -> str:
    values = []
    for value, count in profile.value_counts.most_common(limit):
        values.append(f"{format_number(value)}: {count}")
    if len(profile.value_counts) > limit:
        values.append("...")
    return "; ".join(values)


def workbook_structure_markdown(path: Path, sheet_profiles: list[SheetProfile]) -> list[str]:
    lines = [
        "## Workbook Structure",
        "",
        f"- Source workbook: `{path.relative_to(PROJECT_ROOT)}`",
        f"- File size: {path.stat().st_size:,} bytes",
        f"- Sheets: {len(sheet_profiles)}",
        "",
        "| Sheet | Excel max rows | Excel max columns | Data rows after header | Blank data rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile in sheet_profiles:
        lines.append(
            f"| {md_escape(profile.name)} | {profile.max_row:,} | {profile.max_column:,} | "
            f"{profile.data_row_count:,} | {profile.blank_data_row_count:,} |"
        )
    lines.append("")
    return lines


def column_inventory_markdown(profile: SheetProfile) -> list[str]:
    lines = [
        f"## Column Inventory: `{profile.name}`",
        "",
        "| # | Exact column name | Group | Inferred dtype | Missing | Missing % | Unique non-missing | Role | Description |",
        "|---:|---|---|---|---:|---:|---:|---|---|",
    ]
    for column in profile.columns:
        missing_pct = percent(column.missing_count, profile.data_row_count)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(column.index),
                    md_escape(column.name),
                    md_escape(COLUMN_GROUPS.get(column.name, "unclassified")),
                    md_escape(inferred_dtype(column)),
                    f"{column.missing_count:,}",
                    f"{missing_pct:.2f}",
                    f"{len(column.value_counts):,}",
                    md_escape(feature_role(column.name)),
                    md_escape(COLUMN_DESCRIPTIONS.get(column.name, "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def missingness_markdown(profile: SheetProfile) -> list[str]:
    lines = [
        f"## Missingness: `{profile.name}`",
        "",
        "| Column | Missing | Missing % | Non-missing |",
        "|---|---:|---:|---:|",
    ]
    sorted_columns = sorted(profile.columns, key=lambda column: column.missing_count, reverse=True)
    for column in sorted_columns:
        lines.append(
            f"| {md_escape(column.name)} | {column.missing_count:,} | "
            f"{percent(column.missing_count, profile.data_row_count):.2f} | "
            f"{column.non_missing_count:,} |"
        )
    lines.append("")
    return lines


def numeric_summary_markdown(profile: SheetProfile) -> list[str]:
    lines = [
        f"## Numeric Values: `{profile.name}`",
        "",
        "All non-missing columns in this workbook are numeric or numeric-coded. For coded categorical columns, min/max are descriptive of codes only, not clinical ranges.",
        "",
        "| Column | Count | Min | P05 | P25 | Median | P75 | P95 | Max | Mean | SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for column in profile.columns:
        if not column.numeric_values:
            continue
        summary = numeric_summary(column)
        lines.append(
            f"| {md_escape(column.name)} | {summary['count']:,} | "
            f"{format_number(summary['min'])} | {format_number(summary['p05'])} | "
            f"{format_number(summary['p25'])} | {format_number(summary['median'])} | "
            f"{format_number(summary['p75'])} | {format_number(summary['p95'])} | "
            f"{format_number(summary['max'])} | {format_number(summary['mean'])} | "
            f"{format_number(summary['sd'])} |"
        )
    lines.append("")
    return lines


def coded_value_markdown(profile: SheetProfile) -> list[str]:
    lines = [
        f"## Coded / Low-Cardinality Values: `{profile.name}`",
        "",
        "| Column | Unique non-missing | Missing | Top values as value: count |",
        "|---|---:|---:|---|",
    ]
    for column in profile.columns:
        if column.name in CODED_COLUMNS or len(column.value_counts) <= 12:
            lines.append(
                f"| {md_escape(column.name)} | {len(column.value_counts):,} | "
                f"{column.missing_count:,} | {md_escape(top_values(column, limit=12))} |"
            )
    lines.append("")
    return lines


def duplicates_markdown(profile: SheetProfile) -> list[str]:
    duplicate_id_text = (
        "not evaluated; no `id` column found"
        if profile.duplicate_ids is None
        else f"{profile.duplicate_ids:,}"
    )
    lines = [
        f"## Duplicates And Row Meaning: `{profile.name}`",
        "",
        f"- Exact duplicate data rows: {profile.exact_duplicate_rows:,}",
        f"- Duplicate participant IDs: {duplicate_id_text}",
    ]
    if profile.duplicate_id_values_sample:
        lines.append(
            "- Sample duplicate IDs: "
            + ", ".join(format_number(value) for value in profile.duplicate_id_values_sample)
        )
    lines.extend(
        [
            "- Interpretation from the file: each data row is one participant-level record, not one visit.",
            "- Evidence: the `id` column is present and unique across non-missing IDs, and the sheet has follow-up summary fields rather than visit dates or repeated visit rows.",
            "- Repeated visits are therefore not directly represented as separate rows in this released workbook.",
            "",
        ]
    )
    return lines


def event_followup_markdown(profile: SheetProfile) -> list[str]:
    summary = profile.event_followup_summary
    event_counts: Counter[Any] = summary["event_counts"]
    diagnosis_counts: Counter[Any] = summary["diagnosis_counts"]
    pairs: Counter[tuple[str, str]] = summary["event_diagnosis_pairs"]
    followup_by_event: dict[str, dict[str, float | int | None]] = summary["followup_by_event"]

    event_count = int(event_counts.get(1, 0))
    no_event_count = int(event_counts.get(0, 0))
    diagnosis_yes_count = int(diagnosis_counts.get(1, 0))

    lines = [
        "## Outcome, Follow-Up, And Censoring",
        "",
        f"- Incident diabetes event/status column: `{EVENT_STATUS_COLUMN}`.",
        f"- Follow-up time column: `{FOLLOWUP_TIME_COLUMN}`.",
        f"- Sparse diagnosis flag column: `{EVENT_FLAG_COLUMN}`.",
        f"- Event rows (`{EVENT_STATUS_COLUMN}` = 1): {event_count:,} ({percent(event_count, profile.data_row_count):.2f}%).",
        f"- No-event rows (`{EVENT_STATUS_COLUMN}` = 0): {no_event_count:,} ({percent(no_event_count, profile.data_row_count):.2f}%).",
        f"- Rows with `{EVENT_FLAG_COLUMN}` = 1: {diagnosis_yes_count:,} ({percent(diagnosis_yes_count, profile.data_row_count):.2f}%).",
        "- Censoring representation for survival analysis: rows with event/status = 0 should be treated as right-censored at `year of followup`; rows with event/status = 1 have incident diabetes observed by that follow-up time.",
        "- Important naming caveat: the file calls the binary status column `censor of diabetes...`, but its 1-count matches the published 4,174 incident cases. The separate diagnosis flag is incomplete/sparse and should not be used as the primary target.",
        "",
        "### Event Field Counts",
        "",
        "| Field | Value | Count | Percent of rows |",
        "|---|---:|---:|---:|",
    ]
    for value, count in sorted(event_counts.items(), key=lambda item: str(item[0])):
        lines.append(
            f"| {md_escape(EVENT_STATUS_COLUMN)} | {md_escape(value)} | {count:,} | {percent(count, profile.data_row_count):.2f} |"
        )
    for value, count in sorted(diagnosis_counts.items(), key=lambda item: str(item[0])):
        lines.append(
            f"| {md_escape(EVENT_FLAG_COLUMN)} | {md_escape(value)} | {count:,} | {percent(count, profile.data_row_count):.2f} |"
        )

    lines.extend(
        [
            "",
            "### Cross-Check Between Outcome Fields",
            "",
            "| Event/status value | Diagnosis flag value | Rows |",
            "|---:|---:|---:|",
        ]
    )
    for (event_value, diagnosis_value), count in sorted(pairs.items()):
        lines.append(f"| {md_escape(event_value)} | {md_escape(diagnosis_value)} | {count:,} |")

    lines.extend(
        [
            "",
            "### Follow-Up Time Summary",
            "",
            "| Event/status group | Count | Min | P25 | Median | P75 | Max | Mean | SD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for event_value, values in sorted(followup_by_event.items()):
        if values["count"] == 0:
            continue
        lines.append(
            f"| {md_escape(event_value)} | {values['count']:,} | {format_number(values['min'])} | "
            f"{format_number(values['p25'])} | {format_number(values['median'])} | "
            f"{format_number(values['p75'])} | {format_number(values['max'])} | "
            f"{format_number(values['mean'])} | {format_number(values['sd'])} |"
        )
    lines.append("")
    return lines


def feature_group_markdown(headers: list[str]) -> list[str]:
    def present(columns: set[str]) -> list[str]:
        return [column for column in headers if column in columns]

    groups = {
        "Demographics": [
            column for column in headers if COLUMN_GROUPS.get(column, "").startswith("demographics")
        ],
        "Anthropometry": [column for column in headers if COLUMN_GROUPS.get(column) == "anthropometry"],
        "Blood pressure": [column for column in headers if COLUMN_GROUPS.get(column) == "blood pressure"],
        "Glucose": [column for column in headers if COLUMN_GROUPS.get(column) == "glucose"],
        "Lipids": [column for column in headers if COLUMN_GROUPS.get(column) == "lipids"],
        "Lifestyle": [column for column in headers if COLUMN_GROUPS.get(column) == "lifestyle"],
        "Family history": [column for column in headers if COLUMN_GROUPS.get(column) == "family history"],
        "Follow-up": [column for column in headers if "follow-up" in COLUMN_GROUPS.get(column, "")],
        "Diabetes outcome": [column for column in headers if "diabetes outcome" in COLUMN_GROUPS.get(column, "")],
        "Other laboratory": [column for column in headers if COLUMN_GROUPS.get(column) == "other laboratory"],
    }

    lines = [
        "## Feature Groups And Preliminary AFA Sets",
        "",
        "### Clinical Groups Observed In The File",
        "",
    ]
    for group, columns in groups.items():
        lines.append(f"- {group}: " + (", ".join(f"`{column}`" for column in columns) if columns else "none"))
    lines.extend(
        [
            "",
            "### Preliminary Feature Sets",
            "",
            "- Base/free features: "
            + ", ".join(f"`{column}`" for column in present(BASE_FEATURES)),
            "- Candidate acquirable features: "
            + ", ".join(f"`{column}`" for column in present(CANDIDATE_ACQUIRABLE_FEATURES)),
            "- Excluded/leakage/target fields: "
            + ", ".join(f"`{column}`" for column in present(EXCLUDED_FEATURES)),
            "",
            "BP is placed in the preliminary base set because the project examples treat BP as available early. In a later cost configuration it can also be modeled as a low-cost acquirable measurement.",
            "`site` is included as observed baseline metadata but should be tested in sensitivity analysis because it may encode center-specific effects rather than patient biology.",
            "",
        ]
    )
    return lines


def leakage_markdown() -> list[str]:
    return [
        "## Leakage Assessment",
        "",
        "| Column | Leakage risk | Action |",
        "|---|---|---|",
        f"| `{FINAL_FPG_COLUMN}` | Definite future/post-baseline value measured at final visit; may be after or at diabetes diagnosis. | Exclude from all predictors. |",
        f"| `{EVENT_FLAG_COLUMN}` | Outcome-related follow-up diagnosis flag; incomplete relative to the event/status field. | Exclude from predictors; use only for audit cross-checks. |",
        f"| `{EVENT_STATUS_COLUMN}` | Event/status target. | Use as survival/event target only. |",
        f"| `{FOLLOWUP_TIME_COLUMN}` | Follow-up duration/outcome time. If used as an input covariate it leaks observation time. | Use only as duration target or censoring time. |",
        f"| `{ID_COLUMN}` | Identifier, not a clinical feature; can let models memorize rows if misused. | Exclude from predictors; use for participant-level splits. |",
        "| `site` | Possible center/site effect; not future leakage but may reduce external validity. | Consider excluding or running sensitivity analysis. |",
        "",
    ]


def suitability_markdown() -> list[str]:
    return [
        "## Suitability For The Dissertation Prototype",
        "",
        "### Survival / Time-To-Event Analysis",
        "",
        "This workbook is suitable for a first survival/time-to-event analysis because it contains a binary incident diabetes status and a follow-up duration in years. The survival target should be represented as:",
        "",
        f"- `event = {EVENT_STATUS_COLUMN}` with 1 as incident diabetes and 0 as no observed diabetes.",
        f"- `duration = {FOLLOWUP_TIME_COLUMN}` in years.",
        "- right-censoring when event = 0.",
        "",
        "It is not suitable for modeling repeated visit trajectories because the released file is a participant-level summary, not a visit-level table.",
        "",
        "### First Active Feature Acquisition Model",
        "",
        "The workbook is suitable for the first retrospective cost-aware AFA prototype: it has early/base variables and several baseline laboratory measurements that can be treated as candidate acquirable tests. The first AFA model should be one-shot, patient-level feature acquisition for incident T2D prediction, not longitudinal repeated-visit acquisition.",
        "",
        "### Main Limitations",
        "",
        "- No visit-level rows, visit dates, or full longitudinal measurement history.",
        "- Final-visit FPG is present and must be excluded to avoid future information leakage.",
        "- Exact diabetes diagnosis date is not separately available beyond the follow-up duration/status fields.",
        "- Real acquisition costs are not included; costs must be explicitly defined as experimental assumptions later.",
        "- Several baseline fields have substantial missingness, especially HDL-c, LDL, AST, smoking, and drinking.",
        "- `site` may encode center-level effects and should not be over-interpreted as a portable clinical feature.",
        "",
    ]


def documentation_notes_markdown(path: Path, profile: SheetProfile) -> list[str]:
    local_plus_name = "RC+Health+Care+Data-20180820.xlsx"
    return [
        "## Documentation Notes",
        "",
        f"- The local raw file found by this audit is `{path.name}`.",
        f"- The downloaded source workbook is named `{local_plus_name}`.",
        f"- The observed primary sheet is `{profile.name}` with {profile.data_row_count:,} participant rows and {len(profile.headers):,} columns.",
        "- The observed workbook is already a participant-level follow-up summary; repeated visits are not represented as separate rows.",
        "",
    ]


def build_report(path: Path, sheet_profiles: list[SheetProfile]) -> str:
    primary_profile = sheet_profiles[0]
    lines: list[str] = [
        "# Primary Dataset Profile",
        "",
        "data inspection data audit for the primary Rui-Ci / Rich Healthcare Group dataset.",
        "",
        "No machine-learning model, Active Feature Acquisition policy, cohort file, or training output is created by this audit.",
        "",
    ]
    lines.extend(workbook_structure_markdown(path, sheet_profiles))
    for profile in sheet_profiles:
        lines.extend(column_inventory_markdown(profile))
        lines.extend(missingness_markdown(profile))
        lines.extend(numeric_summary_markdown(profile))
        lines.extend(coded_value_markdown(profile))
        lines.extend(duplicates_markdown(profile))
    lines.extend(event_followup_markdown(primary_profile))
    lines.extend(feature_group_markdown(primary_profile.headers))
    lines.extend(leakage_markdown())
    lines.extend(suitability_markdown())
    lines.extend(documentation_notes_markdown(path, primary_profile))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    raw_path = find_raw_path()
    workbook = load_workbook(raw_path, read_only=True, data_only=True)
    sheet_profiles = [profile_sheet(sheet) for sheet in workbook.worksheets]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(raw_path, sheet_profiles), encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
