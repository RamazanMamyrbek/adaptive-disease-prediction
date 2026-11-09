from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from t2d_aaf.features.reporting import run_feature_cost_design


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate feature definitions and relative acquisition costs.')
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "configs" / "features.yaml")
    parser.add_argument("--costs", type=Path, default=PROJECT_ROOT / "configs" / "costs.yaml")
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
        "--output-manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "feature_cost_design_manifest.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "feature_cost_design.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_feature_cost_design(
        project_root=PROJECT_ROOT,
        feature_config_path=args.features,
        cost_config_path=args.costs,
        cohort_path=args.cohort,
        cohort_manifest_path=args.cohort_manifest,
        output_manifest_path=args.output_manifest,
        output_report_path=args.output_report,
    )
    roles = manifest["roles"]
    print("feature and cost design feature/cost design validated.")
    print(f"Base features: {len(roles['base'])}")
    print(f"Core acquirable features: {len(roles['acquirable'])}")
    print(f"Cost schemes: {len(manifest['costs']['schemes'])}")
    print(f"Default scheme: {manifest['costs']['default_scheme']}")
    print(f"Report: {args.output_report}")
    print(f"Manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
