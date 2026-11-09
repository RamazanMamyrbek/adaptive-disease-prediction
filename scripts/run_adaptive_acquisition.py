from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from t2d_aaf.acquisition.adaptive import (
    DEFAULT_LAMBDA_QUANTILES,
    DEFAULT_RETENTION_TARGET,
    TRAIN_STATE_SEED,
    VALUE_MODEL_ALPHA,
)
from t2d_aaf.acquisition.adaptive_reporting import (
    run_adaptive_acquisition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train and evaluate the adaptive acquisition policy.')
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
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "models" / "adaptive_value_policy.joblib",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "adaptive_acquisition_metrics.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "adaptive_acquisition_results.md",
    )
    parser.add_argument("--retention-target", type=float, default=DEFAULT_RETENTION_TARGET)
    parser.add_argument("--lambda-quantiles", type=int, default=DEFAULT_LAMBDA_QUANTILES)
    parser.add_argument("--train-state-seed", type=int, default=TRAIN_STATE_SEED)
    parser.add_argument("--ridge-alpha", type=float, default=VALUE_MODEL_ALPHA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_adaptive_acquisition(
        project_root=PROJECT_ROOT,
        cohort_path=args.cohort,
        cohort_manifest_path=args.cohort_manifest,
        feature_config_path=args.feature_config,
        cost_config_path=args.cost_config,
        static_metrics_path=args.static_metrics,
        static_model_path=args.static_model,
        ordered_metrics_path=args.ordered_metrics,
        adaptive_model_path=args.model_output,
        metrics_path=args.metrics_output,
        report_path=args.report_output,
        retention_target=args.retention_target,
        lambda_quantiles=args.lambda_quantiles,
        train_state_seed=args.train_state_seed,
        ridge_alpha=args.ridge_alpha,
    )
    selected = results["adaptive_policy"]["selected_operating_point"]
    test = results["evaluation"]["selected_policy"]["test"]
    print("Adaptive acquisition evaluation completed.")
    print(f"Selected cost penalty: {selected['cost_penalty']:.10f}")
    print(
        f"test: C-index={test['prediction']['harrell_c_index']:.4f}, "
        f"IPCW Brier@3y={test['prediction']['ipcw_brier_at_horizon']:.4f}, "
        f"mean tests={test['acquisition']['mean_acquired_tests']:.3f}, "
        f"unique paths={test['acquisition']['unique_paths']}"
    )
    print(f"Report: {args.report_output}")
    print(f"Metrics: {args.metrics_output}")
    print(f"Local policy model: {args.model_output}")


if __name__ == "__main__":
    main()
