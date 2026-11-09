from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2d_aaf.acquisition.adaptive import (
    load_adaptive_model_bundle, prepare_adaptive_data, simulate_adaptive_policy,
    subset_to_mask,
)
from t2d_aaf.evaluation.cost_performance import paired_bootstrap_evaluation, prediction_evaluation
from t2d_aaf.external.nagala import COMPATIBLE_ACQUIRABLE_FEATURES, load_nagala_cohort
from t2d_aaf.external.validation import load_adaptive_value_models
from t2d_aaf.features.registry import load_feature_cost_design
from t2d_aaf.survival.baseline import load_cohort_splits


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def check_reproduction(actual: dict, expected: dict) -> None:
    for metric in ("harrell_c_index", "ipcw_brier_at_horizon"):
        if not np.isclose(actual[metric], expected[metric], rtol=0, atol=1e-12):
            raise ValueError(f"Archived result not reproduced: {metric}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Complete the frozen Base + FPG comparison without fitting or policy tuning.')
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/fpg_comparison_rerun")
    args = parser.parse_args()
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    if out.exists():
        raise ValueError("Use a new output directory; archived experiments are immutable.")
    config_path = root / "configs/fpg_comparison.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["refit_models"] or config["retune_policy"] or config["participant_level_output"]:
        raise ValueError("This script supports frozen aggregate-only evaluation.")
    inputs = ["configs/fpg_comparison.json", "configs/features.yaml", "configs/costs.yaml",
              "data/processed/primary_cohort.csv", "data/processed/primary_cohort_manifest.json",
              "data/raw/RC+Health+Care+Data-20180820.xlsx", "data/external/Data set Int J Obesity.xlsx",
              "models/static_subsets.joblib", "models/adaptive_value_policy.joblib",
              "reports/static_baselines_metrics.json", "reports/adaptive_acquisition_metrics.json",
              "reports/cost_performance_metrics.json", "reports/external_validation_metrics.json"]
    hashes = {p: digest(root / p) for p in inputs}
    read_json = lambda p: json.loads((root / p).read_text(encoding="utf-8"))
    static_results = read_json("reports/static_baselines_metrics.json")
    adaptive_results = read_json("reports/adaptive_acquisition_metrics.json")
    evaluation_results = read_json("reports/cost_performance_metrics.json")
    external_results = read_json("reports/external_validation_metrics.json")
    if static_results["dataset"]["cohort_sha256"] != hashes["data/processed/primary_cohort.csv"]:
        raise ValueError("Cohort differs from the model-development cohort.")
    if adaptive_results["adaptive_model_artifact"]["sha256"] != hashes["models/adaptive_value_policy.joblib"]:
        raise ValueError("Adaptive policy checksum changed.")
    splits = load_cohort_splits(root / "data/processed/primary_cohort.csv")
    nagala = load_nagala_cohort(root / "data/external/Data set Int J Obesity.xlsx")
    bundle = load_adaptive_model_bundle(root / "reports/static_baselines_metrics.json", root / "models/static_subsets.joblib")
    values = load_adaptive_value_models(root / "models/adaptive_value_policy.joblib")
    costs = load_feature_cost_design(root / "configs/features.yaml", root / "configs/costs.yaml").costs
    penalty = adaptive_results["adaptive_policy"]["selected_operating_point"]["cost_penalty"]
    horizon = config["horizon_years"]
    results = {}
    started = time.monotonic()
    for name, frame, censoring_frame, available in [
        ("internal", splits.test, splits.train, None),
        ("external", nagala.mapped, nagala.mapped, COMPATIBLE_ACQUIRABLE_FEATURES),
    ]:
        print(f"{name}: frozen predictions", flush=True)
        prepared = prepare_adaptive_data(bundle, frame, horizon=horizon)
        kwargs = {} if available is None else {"available_candidates": available}
        adaptive = simulate_adaptive_policy(prepared, values, costs, penalty, **kwargs)
        risks = {"base_plus_fpg": prepared.subset_risks[:, subset_to_mask(("fpg",))],
                 "adaptive": adaptive.selected_risk}
        per_cost = {"base_plus_fpg": np.ones(len(frame)), "adaptive": adaptive.acquired_count.astype(float)}
        t, d = frame["duration_years"].to_numpy(float), frame["event"].to_numpy(np.int64)
        ct, cd = censoring_frame["duration_years"].to_numpy(float), censoring_frame["event"].to_numpy(np.int64)
        metrics = {key: prediction_evaluation(ct, cd, t, d, risk, horizon) for key, risk in risks.items()}
        expected = (evaluation_results["primary_comparison"]["adaptive_acquisition"]["test"]["prediction"]
                    if name == "internal" else external_results["methods"]["adaptive_compatible"]["prediction"])
        check_reproduction(metrics["adaptive"], expected)
        if name == "internal":
            check_reproduction(metrics["base_plus_fpg"], next(x for x in static_results["all_subset_results"] if x["subset_id"] == "fpg")["metrics"]["test"])
        print(f"{name}: {config['bootstrap_replicates']} paired bootstrap replicates", flush=True)
        bootstrap = paired_bootstrap_evaluation(ct, cd, t, d, risks, per_cost, horizon,
            contrasts=tuple(tuple(x) for x in config["contrasts"]),
            replicates=config["bootstrap_replicates"], seed=config["bootstrap_seed"])
        results[name] = {"methods": {key: {"prediction": metrics[key],
                            "mean_cost": float(np.mean(per_cost[key]))} for key in risks},
                         "bootstrap": bootstrap, "archived_adaptive_reproduced": True,
                         "censoring_source": "primary training" if name == "internal" else "NAGALA evaluation cohort"}
        print(json.dumps({"cohort": name, "fpg_c": metrics["base_plus_fpg"]["harrell_c_index"],
                          "contrast": bootstrap["contrasts"]["adaptive_minus_base_plus_fpg"]}), flush=True)
    unchanged = hashes == {p: digest(root / p) for p in inputs}
    if not unchanged:
        raise ValueError("An input changed during evaluation.")
    out.mkdir(parents=True)
    document = {"schema": "fpg_comparison_v1", "protocol": config, "results": results,
                "input_sha256": hashes, "script_sha256": digest(Path(__file__)), "inputs_unchanged": unchanged,
                "models_fitted": False, "policy_retuned": False, "participant_level_output": None,
                "python": platform.python_version(), "elapsed_seconds": time.monotonic() - started}
    (out / "metrics.json").write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Completed aggregate-only comparison: {out}", flush=True)


if __name__ == "__main__":
    main()
