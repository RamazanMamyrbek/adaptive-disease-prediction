from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from t2d_aaf.acquisition.adaptive import DEFAULT_LAMBDA_QUANTILES
from t2d_aaf.evaluation.cost_performance_reporting import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    EVALUATION_BOOTSTRAP_SEED,
    PRIMARY_RETENTION_TARGET,
    run_cost_performance_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate predictive performance and acquisition cost.')
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
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "cost_performance_metrics.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "cost_performance_evaluation.md",
    )
    parser.add_argument(
        "--cost-figure-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "cost_performance_uniform.png",
    )
    parser.add_argument(
        "--calibration-figure-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "calibration_test.png",
    )
    parser.add_argument(
        "--retention-target", type=float, default=PRIMARY_RETENTION_TARGET
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=EVALUATION_BOOTSTRAP_SEED)
    parser.add_argument("--lambda-quantiles", type=int, default=DEFAULT_LAMBDA_QUANTILES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_cost_performance_evaluation(
        project_root=PROJECT_ROOT,
        cohort_path=args.cohort,
        cohort_manifest_path=args.cohort_manifest,
        feature_config_path=args.feature_config,
        cost_config_path=args.cost_config,
        static_metrics_path=args.static_metrics,
        static_model_path=args.static_model,
        ordered_metrics_path=args.ordered_metrics,
        adaptive_metrics_path=args.adaptive_metrics,
        adaptive_model_path=args.adaptive_model,
        metrics_path=args.metrics_output,
        report_path=args.report_output,
        cost_figure_path=args.cost_figure_output,
        calibration_figure_path=args.calibration_figure_output,
        retention_target=args.retention_target,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        lambda_quantiles=args.lambda_quantiles,
    )
    adaptive = results["primary_comparison"]["adaptive_acquisition"]["test"]
    ordered = results["primary_comparison"]["ordered_acquisition"]["test"]
    print("Cost-performance evaluation completed.")
    print(
        "Adaptive test: "
        f"C-index={adaptive['prediction']['harrell_c_index']:.4f}, "
        f"IPCW AUC@3y={adaptive['prediction']['ipcw_cumulative_dynamic_auc']:.4f}, "
        f"IPCW PR-AUC@3y={adaptive['prediction']['ipcw_average_precision']:.4f}, "
        f"mean cost={adaptive['acquisition']['mean_cost']:.3f}"
    )
    print(
        "Ordered test: "
        f"C-index={ordered['prediction']['harrell_c_index']:.4f}, "
        f"mean cost={ordered['acquisition']['mean_cost']:.3f}"
    )
    print(f"Report: {args.report_output}")
    print(f"Metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
