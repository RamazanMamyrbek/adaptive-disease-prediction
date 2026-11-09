from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from t2d_aaf.data.cohort import (
    DEFAULT_SEED,
    build_cohort_report,
    build_primary_cohort,
    find_primary_raw_path,
    try_write_parquet,
    write_csv,
    write_manifest,
    write_report,
)


CSV_PATH = PROJECT_ROOT / "data" / "processed" / "primary_cohort.csv"
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "primary_cohort.parquet"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "primary_cohort_manifest.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "cohort_summary.md"


def main() -> None:
    raw_path = find_primary_raw_path(PROJECT_ROOT)
    result = build_primary_cohort(raw_path, seed=DEFAULT_SEED)
    write_csv(result.records, CSV_PATH)

    parquet_status = try_write_parquet(result.records, PARQUET_PATH)
    result = type(result)(
        records=result.records,
        raw_row_count=result.raw_row_count,
        row_exclusions=result.row_exclusions,
        duplicate_participant_ids=result.duplicate_participant_ids,
        seed=result.seed,
        split_fractions=result.split_fractions,
        parquet_status=parquet_status,
    )

    write_manifest(
        result,
        raw_path=raw_path,
        csv_path=CSV_PATH,
        parquet_path=PARQUET_PATH,
        manifest_path=MANIFEST_PATH,
    )
    report_text = build_cohort_report(
        result,
        raw_path=raw_path,
        csv_path=CSV_PATH,
        parquet_path=PARQUET_PATH,
        manifest_path=MANIFEST_PATH,
    )
    write_report(report_text, REPORT_PATH)

    event_count = sum(int(record["event"]) for record in result.records)
    print(f"Wrote {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    if parquet_status.get("written"):
        print(f"Wrote {PARQUET_PATH.relative_to(PROJECT_ROOT)}")
    else:
        print(f"Parquet not written: {parquet_status.get('reason')}")
    print(f"Cohort participants: {len(result.records):,}; events: {event_count:,}")


if __name__ == "__main__":
    main()
