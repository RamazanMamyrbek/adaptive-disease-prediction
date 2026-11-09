from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from t2d_aaf.survival.reporting import run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train and evaluate the Cox survival baseline.')
    parser.add_argument(
        "--cohort",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "primary_cohort.csv",
    )
    parser.add_argument(
        "--cohort-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "primary_cohort_manifest.json",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "models" / "cox_baseline.joblib",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "ordinary_baseline_metrics.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "ordinary_baseline_results.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_baseline(
        project_root=PROJECT_ROOT,
        cohort_path=args.cohort,
        cohort_manifest_path=args.cohort_manifest,
        model_path=args.model_output,
        metrics_path=args.metrics_output,
        report_path=args.report_output,
    )
    metrics = results["evaluation"]["metrics"]
    print("Cox baseline evaluation completed.")
    for split in ("train", "validation", "test"):
        row = metrics[split]
        print(
            f"{split}: n={row['participants']:,}, events={row['events_total']:,}, "
            f"C-index={row['harrell_c_index']:.4f}, "
            f"IPCW Brier@3y={row['ipcw_brier_at_horizon']:.4f}"
        )
    print(f"Report: {args.report_output}")
    print(f"Metrics: {args.metrics_output}")
    print(f"Local model artifact: {args.model_output}")


if __name__ == "__main__":
    main()
