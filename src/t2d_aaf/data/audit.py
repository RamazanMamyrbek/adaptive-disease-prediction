import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import openpyxl


HORIZONS = (2.0, 3.0, 4.0)
RAW_COLUMNS = {
    "id": "id",
    "event": "censor of diabetes at followup(1, Yes; 0, No)",
    "duration": "year of followup",
    "fpg": "FPG (mmol/L)",
    "final_fpg": "FPG of final visit(mmol/L)",
    "diagnosis": "Diabetes diagnosed during followup\uff081,Yes\uff09",
}
BASE = ("age", "gender", "bmi", "sbp", "dbp", "family_history_diabetes")
LABS = ("fpg", "total_cholesterol", "triglycerides", "alt", "ccr", "bun")


def number(value, *, missing_ok=False):
    if value is None or (isinstance(value, str) and not value.strip()):
        if missing_ok:
            return None
        raise ValueError("Missing required numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite value")
    return result


def target(event, duration):
    event, duration = number(event), number(duration)
    if event not in (0, 1) or duration <= 0:
        raise ValueError("Invalid survival target")
    return int(event), duration


def horizon_state(event, duration, horizon):
    event, duration = target(event, duration)
    horizon = number(horizon)
    if horizon <= 0:
        raise ValueError("Horizon must be positive")
    if duration > horizon:
        return "known_event_free_at_h"
    if event == 1:
        return "event_by_h"
    return "censored_at_or_before_h"


def composite_evidence(final_fpg, diagnosis):
    final_fpg = number(final_fpg, missing_ok=True)
    diagnosis = number(diagnosis, missing_ok=True)
    if diagnosis not in (None, 0, 1):
        raise ValueError("Unexpected diagnosis flag")
    if diagnosis == 1 or (final_fpg is not None and final_fpg >= 7):
        return "positive_component"
    if final_fpg is None:
        return "insufficient_components"
    return "no_positive_component"


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def summary(values):
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.fmean(values), "max": max(values), "sum": math.fsum(values)}


def run(project, output):
    raw = project / "data/raw/RC+Health+Care+Data-20180820.xlsx"
    cohort = project / "data/processed/primary_cohort.csv"
    manifest = project / "data/processed/primary_cohort_manifest.json"
    inputs = (raw, cohort, manifest)
    output = output.resolve()
    for tree in (project / "data/raw", project / "data/processed"):
        if output.is_relative_to(tree.resolve()):
            raise ValueError("Audit output cannot be inside a data directory")
    if output in {path.resolve() for path in inputs}:
        raise ValueError("Output would overwrite an input")
    before = {str(path.relative_to(project)): digest(path) for path in inputs}
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    by_id, split_counts, missing = {}, defaultdict(Counter), Counter()
    durations, horizon_counts = defaultdict(list), defaultdict(Counter)
    ties, sites, violations = Counter(), Counter(), Counter()
    fpg_values = []
    with cohort.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        required = set(BASE + LABS + ("participant_id", "split", "event", "duration_years", "site"))
        if not required <= set(columns or []):
            raise ValueError("Cohort schema mismatch")
        for row in reader:
            pid = row["participant_id"]
            if not pid:
                raise ValueError("Missing participant identifier")
            if pid in by_id:
                violations["duplicate_processed_id"] += 1
            event, duration = target(row["event"], row["duration_years"])
            split = row["split"]
            if split not in ("train", "validation", "test"):
                raise ValueError("Unexpected split")
            fpg = number(row["fpg"], missing_ok=True)
            by_id[pid] = (event, duration, fpg)
            split_counts[split].update({"n": 1, "events_all_followup": event})
            sites[row["site"]] += 1
            if fpg is not None:
                fpg_values.append(fpg)
                violations["baseline_fpg_ge_7"] += int(fpg >= 7)
            for name in BASE + LABS:
                missing[name] += int(number(row[name], missing_ok=True) is None)
            for group in ("all", f"event_{event}", f"split_{split}"):
                durations[group].append(duration)
            for horizon in HORIZONS:
                state = horizon_state(event, duration, horizon)
                for group in ("all", split):
                    horizon_counts[f"{group}@{horizon:g}"].update({state: 1})
                if duration == horizon:
                    ties[f"h={horizon:g},event={event}"] += 1
    print("Processed cohort read; auditing raw workbook locally.", flush=True)
    workbook = openpyxl.load_workbook(raw, read_only=True, data_only=True)
    raw_counts, cross_tab, raw_seen = Counter(), Counter(), set()
    sheets = workbook.sheetnames
    try:
        rows = workbook["RC"].iter_rows(values_only=True)
        header = next(rows)
        if any(name not in header for name in RAW_COLUMNS.values()):
            raise ValueError("Raw schema mismatch")
        indices = {name: header.index(label) for name, label in RAW_COLUMNS.items()}
        for row in rows:
            if all(value is None for value in row):
                continue
            value = {key: row[index] for key, index in indices.items()}
            numeric_id = number(value["id"])
            if not numeric_id.is_integer():
                raise ValueError("Non-integer raw identifier")
            pid = str(int(numeric_id))
            raw_counts["n"] += 1
            if pid in raw_seen:
                violations["duplicate_raw_id"] += 1
            raw_seen.add(pid)
            event, duration = target(value["event"], value["duration"])
            fpg = number(value["fpg"], missing_ok=True)
            final_fpg = number(value["final_fpg"], missing_ok=True)
            diagnosis = number(value["diagnosis"], missing_ok=True)
            evidence = composite_evidence(final_fpg, diagnosis)
            category = "missing" if final_fpg is None else ("ge7" if final_fpg >= 7 else "lt7")
            flag = "missing" if diagnosis is None else str(int(diagnosis))
            cross_tab[(event, category, flag)] += 1
            raw_counts[f"event_{event}__{evidence}"] += 1
            raw_counts["events_all_followup"] += event
            if evidence == "positive_component" and event == 0:
                violations["positive_component_but_event0"] += 1
            if evidence == "no_positive_component" and event == 1:
                violations["event1_without_observed_positive_component"] += 1
            if pid not in by_id:
                violations["raw_id_not_in_processed"] += 1
            else:
                processed = by_id[pid]
                if event != processed[0]:
                    violations["event_mismatch"] += 1
                if not math.isclose(duration, processed[1], rel_tol=0, abs_tol=1e-9):
                    violations["duration_mismatch_gt_1e-9"] += 1
                if (fpg is None) != (processed[2] is None) or (fpg is not None and
                        not math.isclose(fpg, processed[2], rel_tol=0, abs_tol=1e-9)):
                    violations["fpg_mismatch_gt_1e-9"] += 1
    finally:
        workbook.close()
    violations["processed_ids_not_in_raw"] = len(set(by_id) - raw_seen)
    violations["manifest_raw_count_mismatch"] = int(metadata["raw_row_count"] != raw_counts["n"])
    violations["manifest_cohort_count_mismatch"] = int(
        metadata["cohort_row_count"] != sum(count["n"] for count in split_counts.values()))
    after = {str(path.relative_to(project)): digest(path) for path in inputs}
    if before != after:
        raise RuntimeError("An input changed during the audit; output not written")
    violation_names = (
        "duplicate_processed_id", "duplicate_raw_id", "baseline_fpg_ge_7",
        "positive_component_but_event0", "event1_without_observed_positive_component",
        "raw_id_not_in_processed", "processed_ids_not_in_raw", "event_mismatch",
        "duration_mismatch_gt_1e-9", "fpg_mismatch_gt_1e-9",
        "manifest_raw_count_mismatch", "manifest_cohort_count_mismatch",
    )
    result = {
        "schema": "read_only_target_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script_sha256": digest(Path(__file__)),
        "runtime": {"python": sys.version.split()[0], "openpyxl": openpyxl.__version__},
        "input_sha256": before, "inputs_unchanged": True,
        "model_training_performed": False, "participant_records_exported": False,
        "processed_columns": columns, "raw_sheets": sheets,
        "processed_n": sum(count["n"] for count in split_counts.values()),
        "unique_processed_ids": len(by_id), "raw_counts": dict(sorted(raw_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "horizon_counts": dict(sorted(horizon_counts.items())), "exact_horizon_ties": dict(ties),
        "duration_summary": {key: summary(value) for key, value in sorted(durations.items())},
        "fpg_summary": summary(fpg_values), "missing_counts": dict(missing),
        "site_code_count": len(sites), "site_sizes": dict(sorted(sites.items())),
        "raw_composite_crosstab": [
            {"event": key[0], "final_fpg": key[1], "diagnosis_flag": key[2], "n": count}
            for key, count in sorted(cross_tab.items())],
        "violations": {key: violations[key] for key in violation_names},
        "component_note": "Missing diagnosis flag is not a verified negative diagnosis history.",
        "horizon_note": "Event at h is a case; censoring at h is unknown under repository convention.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"processed_n": result["processed_n"], "raw_counts": result["raw_counts"],
                      "violations": result["violations"], "inputs_unchanged": True}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Read-only target audit. Persist aggregate counts, never participant records.')
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit.json"))
    args = parser.parse_args()
    run(args.project.resolve(), args.output)
