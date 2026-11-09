from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from t2d_aaf.acquisition.ordered import (
    DEFAULT_RETENTION_TARGET,
    DEFAULT_THRESHOLD_QUANTILES,
)
from t2d_aaf.acquisition.ordered_reporting import (
    run_ordered_acquisition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate the fixed-order NEXT/STOP acquisition baseline.')
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
        "--feature-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "features.yaml",
    )
    parser.add_argument(
        "--cost-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "costs.yaml",
    )
    parser.add_argument(
        "--static-metrics",
        type=Path,
        default=PROJECT_ROOT / "reports" / "static_baselines_metrics.json",
    )
    parser.add_argument(
        "--static-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "static_subsets.joblib",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "ordered_acquisition_metrics.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "ordered_acquisition_results.md",
    )
    parser.add_argument("--retention-target", type=float, default=DEFAULT_RETENTION_TARGET)
    parser.add_argument(
        "--threshold-quantiles",
        type=int,
        default=DEFAULT_THRESHOLD_QUANTILES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_ordered_acquisition(
        project_root=PROJECT_ROOT,
        cohort_path=args.cohort,
        cohort_manifest_path=args.cohort_manifest,
        feature_config_path=args.feature_config,
        cost_config_path=args.cost_config,
        static_metrics_path=args.static_metrics,
        static_model_path=args.static_model,
        metrics_path=args.metrics_output,
        report_path=args.report_output,
        retention_target=args.retention_target,
        threshold_quantiles=args.threshold_quantiles,
    )
    selected = results["ordered_policy"]["selected_operating_point"]
    test = results["evaluation"]["selected_policy"]["test"]
    print("Ordered acquisition evaluation completed.")
    print("Order: " + " -> ".join(results["fixed_order"]["order"]))
    print(f"Selected threshold: {selected['threshold']:.8f}")
    print(
        f"test: C-index={test['prediction']['harrell_c_index']:.4f}, "
        f"IPCW Brier@3y={test['prediction']['ipcw_brier_at_horizon']:.4f}, "
        f"mean tests={test['acquisition']['mean_acquired_tests']:.3f}"
    )
    print(f"Report: {args.report_output}")
    print(f"Metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
