from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from t2d_aaf.external.reporting import (
    DEFAULT_EXTERNAL_BOOTSTRAP_REPLICATES,
    EXTERNAL_BOOTSTRAP_SEED,
    run_external_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate compatible features on the external NAGALA cohort.')
    parser.add_argument(
        "--external-data",
        type=Path,
        default=PROJECT_ROOT / "data" / "external" / "Data set Int J Obesity.xlsx",
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
        "--ordered-metrics",
        type=Path,
        default=PROJECT_ROOT / "reports" / "ordered_acquisition_metrics.json",
    )
    parser.add_argument(
        "--adaptive-metrics",
        type=Path,
        default=PROJECT_ROOT / "reports" / "adaptive_acquisition_metrics.json",
    )
    parser.add_argument(
        "--adaptive-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "adaptive_value_policy.joblib",
    )
    parser.add_argument(
        "--evaluation-metrics",
        type=Path,
        default=PROJECT_ROOT / "reports" / "cost_performance_metrics.json",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "nagala_dataset_profile.md",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "external_validation_metrics.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "external_validation_results.md",
    )
    parser.add_argument(
        "--cost-figure-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "external_cost_performance.png",
    )
    parser.add_argument(
        "--calibration-figure-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "external_calibration.png",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_EXTERNAL_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=EXTERNAL_BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_external_validation(
        project_root=PROJECT_ROOT,
        external_path=args.external_data,
        feature_config_path=args.feature_config,
        cost_config_path=args.cost_config,
        static_metrics_path=args.static_metrics,
        static_model_path=args.static_model,
        ordered_metrics_path=args.ordered_metrics,
        adaptive_metrics_path=args.adaptive_metrics,
        adaptive_model_path=args.adaptive_model,
        evaluation_metrics_path=args.evaluation_metrics,
        profile_path=args.profile_output,
        metrics_path=args.metrics_output,
        report_path=args.report_output,
        cost_figure_path=args.cost_figure_output,
        calibration_figure_path=args.calibration_figure_output,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    adaptive = results["methods"]["adaptive_compatible"]
    ordered = results["methods"]["ordered_compatible"]
    print("NAGALA external validation completed.")
    print(
        "Adaptive compatible: "
        f"C-index={adaptive['prediction']['harrell_c_index']:.4f}, "
        f"IPCW AUC@3y={adaptive['prediction']['ipcw_cumulative_dynamic_auc']:.4f}, "
        f"mean cost={adaptive['acquisition']['mean_cost']:.3f}"
    )
    print(
        "Ordered compatible: "
        f"C-index={ordered['prediction']['harrell_c_index']:.4f}, "
        f"mean cost={ordered['acquisition']['mean_cost']:.3f}"
    )
    print(f"Dataset profile: {args.profile_output}")
    print(f"Validation report: {args.report_output}")
    print(f"Metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
