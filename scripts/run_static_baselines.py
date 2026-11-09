from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from t2d_aaf.survival.static_baselines import (
    DEFAULT_RANDOM_REPEATS,
    STATIC_RANDOM_SEED,
)
from t2d_aaf.survival.static_reporting import run_static_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fit and evaluate static feature-set baselines.')
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
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "models" / "static_subsets.joblib",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "static_baselines_metrics.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "static_baselines_results.md",
    )
    parser.add_argument("--random-repeats", type=int, default=DEFAULT_RANDOM_REPEATS)
    parser.add_argument("--random-seed", type=int, default=STATIC_RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_static_baselines(
        project_root=PROJECT_ROOT,
        cohort_path=args.cohort,
        cohort_manifest_path=args.cohort_manifest,
        feature_config_path=args.feature_config,
        cost_config_path=args.cost_config,
        model_path=args.model_output,
        metrics_path=args.metrics_output,
        report_path=args.report_output,
        random_repeats=args.random_repeats,
        random_seed=args.random_seed,
    )
    print("Static baseline evaluation completed.")
    print("Top-K order: " + " -> ".join(results["methods"]["static_top_k"]["order"]))
    print(
        "Greedy order: "
        + " -> ".join(results["methods"]["importance_greedy"]["order"])
    )
    print(f"Report: {args.report_output}")
    print(f"Metrics: {args.metrics_output}")
    print(f"Local model artifact: {args.model_output}")


if __name__ == "__main__":
    main()
