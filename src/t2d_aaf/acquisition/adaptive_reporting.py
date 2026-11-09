from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

from t2d_aaf.acquisition.adaptive import (
    DEFAULT_LAMBDA_QUANTILES,
    DEFAULT_RETENTION_TARGET,
    POLICY_NAME,
    TRAIN_STATE_SEED,
    VALUE_MODEL_ALPHA,
    VALUE_MODEL_TYPE,
    FittedAdaptiveValueModels,
    LambdaSearchPoint,
    SelectedAdaptiveOperatingPoint,
    adaptive_acquisition_summary,
    adaptive_prediction_metrics,
    evaluate_lambda_grid,
    fit_adaptive_value_models,
    load_adaptive_model_bundle,
    prepare_adaptive_data,
    save_adaptive_value_artifact,
    select_adaptive_operating_point,
    simulate_adaptive_policy,
)
from t2d_aaf.data.cohort import (
    BASE_FEATURES,
    CORE_ACQUIRABLE_FEATURES,
    DEFAULT_SEED,
    EXCLUDED_FROM_MAIN_PREDICTORS,
    OPTIONAL_FEATURES,
)
from t2d_aaf.features.registry import load_feature_cost_design
from t2d_aaf.survival.baseline import EVALUATION_HORIZON_YEARS, load_cohort_splits
from t2d_aaf.survival.reporting import package_versions, portable_path, sha256_file


def _penalty_json(value: float) -> float | None:
    return None if math.isinf(value) else float(value)


def _search_point_json(point: LambdaSearchPoint) -> dict[str, float | int | None]:
    output = asdict(point)
    output["cost_penalty"] = _penalty_json(point.cost_penalty)
    return output


def _selected_json(selected: SelectedAdaptiveOperatingPoint) -> dict[str, Any]:
    return {
        "cost_penalty": _penalty_json(selected.point.cost_penalty),
        "retention_target": selected.retention_target,
        "target_validation_c_index": selected.target_validation_c_index,
        "base_validation_c_index": selected.base_validation_c_index,
        "full_validation_c_index": selected.full_validation_c_index,
        "selected_validation_c_index": selected.point.validation_c_index,
        "selected_validation_ipcw_brier": selected.point.validation_ipcw_brier,
        "retained_improvement_fraction": selected.retained_improvement_fraction,
        "mean_acquired_tests": selected.point.mean_acquired_tests,
        "median_acquired_tests": selected.point.median_acquired_tests,
        "mean_primary_cost": selected.point.mean_primary_cost,
        "stop_without_labs_percent": selected.point.stop_without_labs_percent,
        "acquire_all_percent": selected.point.acquire_all_percent,
        "unique_paths": selected.point.unique_paths,
        "unique_final_subsets": selected.point.unique_final_subsets,
    }


def _validate_provenance(
    static_results: Mapping[str, Any],
    ordered_results: Mapping[str, Any],
    cohort_path: Path,
    selected: SelectedAdaptiveOperatingPoint,
) -> None:
    cohort_hash = sha256_file(cohort_path)
    if static_results["dataset"]["cohort_sha256"] != cohort_hash:
        raise ValueError("static baseline metrics reference a different cohort.")
    if ordered_results.get("stage") != 6:
        raise ValueError("Adaptive reporting requires ordered acquisition comparator metrics.")
    if ordered_results["dataset"]["cohort_sha256"] != cohort_hash:
        raise ValueError("ordered acquisition metrics reference a different cohort.")
    if ordered_results["tuning"]["test_used_for_threshold_selection"] is not False:
        raise ValueError("ordered acquisition comparator must preserve its test lock.")
    static_base = float(
        static_results["methods"]["base_only"]["result"]["metrics"]["validation"][
            "harrell_c_index"
        ]
    )
    static_full = float(
        static_results["methods"]["full_features"]["result"]["metrics"]["validation"][
            "harrell_c_index"
        ]
    )
    if not math.isclose(
        selected.base_validation_c_index, static_base, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Adaptive always-STOP endpoint differs from static baseline Base Only.")
    if not math.isclose(
        selected.full_validation_c_index, static_full, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Adaptive always-acquire endpoint differs from static baseline Full Features.")


def build_adaptive_results(
    *,
    project_root: Path,
    cohort_path: Path,
    cohort_manifest_path: Path,
    feature_config_path: Path,
    cost_config_path: Path,
    static_metrics_path: Path,
    static_model_path: Path,
    ordered_metrics_path: Path,
    adaptive_model_path: Path,
    static_results: Mapping[str, Any],
    ordered_results: Mapping[str, Any],
    value_models: FittedAdaptiveValueModels,
    search_points: tuple[LambdaSearchPoint, ...],
    selected: SelectedAdaptiveOperatingPoint,
    lambda_quantiles: int,
    split_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    design = load_feature_cost_design(feature_config_path, cost_config_path)
    always_acquire = next(point for point in search_points if point.cost_penalty == 0.0)
    always_stop = next(point for point in search_points if math.isinf(point.cost_penalty))
    return {
        "schema_version": "adaptive_acquisition_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 7,
        "scope": "free-choice patient-specific laboratory acquisition plus STOP",
        "seeds": {
            "cohort_split": DEFAULT_SEED,
            "train_state_sampling": value_models.seed,
            "policy_randomness_at_inference": None,
        },
        "dataset": {
            "cohort_path": portable_path(cohort_path, project_root),
            "cohort_sha256": sha256_file(cohort_path),
            "cohort_manifest_path": portable_path(cohort_manifest_path, project_root),
            "cohort_manifest_sha256": sha256_file(cohort_manifest_path),
        },
        "static_provenance": {
            "metrics_path": portable_path(static_metrics_path, project_root),
            "metrics_sha256": sha256_file(static_metrics_path),
            "local_subset_model_path": portable_path(static_model_path, project_root),
            "local_subset_model_sha256": sha256_file(static_model_path),
            "models_fit_split": static_results["model"]["fit_split"],
            "preprocessing_fit_split": static_results["preprocessing"]["fit_split"],
            "subset_models_reused": 64,
        },
        "ordered_comparator": {
            "metrics_path": portable_path(ordered_metrics_path, project_root),
            "metrics_sha256": sha256_file(ordered_metrics_path),
            "used_for_adaptive_training_or_selection": False,
        },
        "adaptive_model_artifact": {
            "local_path": portable_path(adaptive_model_path, project_root),
            "sha256": sha256_file(adaptive_model_path),
            "committed": False,
        },
        "target": {"event": "event", "duration": "duration_years", "censoring": "event = 0"},
        "features": {
            "base": list(BASE_FEATURES),
            "acquisition_candidates": list(CORE_ACQUIRABLE_FEATURES),
            "optional_not_used": list(OPTIONAL_FEATURES),
            "excluded_from_predictors_and_policy_state": list(
                EXCLUDED_FROM_MAIN_PREDICTORS
            ),
        },
        "state_contract": {
            "state_feature_names": list(value_models.state_feature_names),
            "observed_values": "base values plus acquired candidate values; unknown slots are zero",
            "acquisition_mask": "one bit per candidate test",
            "source_observation_mask": "separate mask for source availability of observed values",
            "current_prediction": "3-year risk from the Cox model for the current subset",
            "available_actions": [*CORE_ACQUIRABLE_FEATURES, "STOP"],
            "candidate_costs": "named relative-cost scheme",
            "outcome_or_followup_available_to_policy": False,
            "identifier_or_site_available_to_policy": False,
            "unacquired_candidate_values_available_to_policy": False,
        },
        "value_models": {
            "type": VALUE_MODEL_TYPE,
            "fit_split": "train",
            "models": len(value_models.models),
            "ridge_alpha": value_models.alpha,
            "training_seed": value_models.seed,
            "sample_strategy": (
                "one deterministic random subset of the other five candidates per "
                "train participant and candidate model"
            ),
            "target": (
                "absolute 3-year risk revision |risk(S union candidate) - risk(S)| "
                "from train-fitted static baseline Cox subset models"
            ),
            "target_uses_event_or_followup_directly": False,
            "candidate_models": [
                {
                    "candidate": model.candidate,
                    **dict(model.training_summary),
                }
                for model in value_models.models
            ],
        },
        "adaptive_policy": {
            "name": POLICY_NAME,
            "action_space": [*CORE_ACQUIRABLE_FEATURES, "STOP"],
            "free_choice_among_remaining_tests": True,
            "patient_specific_test_order": True,
            "predicted_benefit": "non-negative clipped Ridge prediction",
            "utility": "predicted absolute risk revision - cost_penalty * candidate cost",
            "decision_rule": (
                "acquire the remaining candidate with maximum non-negative utility; "
                "choose STOP when every remaining utility is negative; force STOP after six"
            ),
            "selected_operating_point": _selected_json(selected),
            "missing_result_rule": (
                "charge the selected action, retain the participant, use the train-fitted "
                "imputation value, and expose source absence through the source mask"
            ),
        },
        "tuning": {
            "cost_penalty_candidate_source": (
                "quantiles of train predicted benefit divided by primary-scheme cost"
            ),
            "lambda_quantile_count_requested": lambda_quantiles,
            "cost_penalties_evaluated": len(search_points),
            "selection_split": "validation",
            "selection_metric": "Harrell C-index retention relative to Base Only and Full Features",
            "selection_rule": (
                "minimum mean primary-scheme cost among candidates retaining the target "
                "fraction of validation C-index improvement"
            ),
            "test_used_for_cost_penalty_selection": False,
            "test_cost_penalties_evaluated": 1,
            "always_stop_endpoint": _search_point_json(always_stop),
            "always_acquire_endpoint": _search_point_json(always_acquire),
            "search": [_search_point_json(point) for point in search_points],
        },
        "evaluation": {
            "horizon_years": EVALUATION_HORIZON_YEARS,
            "brier": "single-horizon IPCW; censoring distribution estimated on train only",
            "selected_policy": dict(split_results),
            "static_references": {
                "base_only": static_results["methods"]["base_only"]["result"]["metrics"],
                "full_features": static_results["methods"]["full_features"]["result"][
                    "metrics"
                ],
            },
            "ordered_reference": ordered_results["evaluation"]["selected_policy"],
        },
        "costs": {
            "config_path": portable_path(cost_config_path, project_root),
            "default_scheme": design.costs.default_scheme,
            "unit_type": design.costs.semantics["unit_type"],
            "real_prices": design.costs.semantics["real_prices"],
            "stop_cost": design.costs.stop_cost,
            "schemes": {
                name: {"role": scheme.role, "costs": dict(scheme.costs)}
                for name, scheme in design.costs.schemes.items()
            },
        },
        "participant_level_output": None,
        "environment": package_versions(),
    }


def write_results_json(results: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def _metric(value: float) -> str:
    return f"{value:.4f}"


def _percent(value: float) -> str:
    return f"{value:.2f}%"


def build_markdown_report(results: Mapping[str, Any]) -> str:
    selected = results["adaptive_policy"]["selected_operating_point"]
    evaluation = results["evaluation"]
    adaptive_validation = evaluation["selected_policy"]["validation"]
    adaptive_test = evaluation["selected_policy"]["test"]
    ordered = evaluation["ordered_reference"]
    base = evaluation["static_references"]["base_only"]
    full = evaluation["static_references"]["full_features"]
    test_acquisition = adaptive_test["acquisition"]
    ordered_test_cost = ordered["test"]["acquisition"]["cost_by_scheme"]["uniform_unit"][
        "mean"
    ]
    adaptive_test_cost = test_acquisition["cost_by_scheme"]["uniform_unit"]["mean"]
    vs_full_reduction = 100.0 * (1.0 - adaptive_test_cost / len(CORE_ACQUIRABLE_FEATURES))
    vs_ordered_reduction = 100.0 * (1.0 - adaptive_test_cost / ordered_test_cost)
    ordered_path_counts = {
        split: sum(
            row["participants"] > 0
            for row in ordered[split]["acquisition"]["stage_distribution"]
        )
        for split in ("validation", "test")
    }

    lines = [
        "# adaptive acquisition Adaptive Acquisition",
        "",
        "adaptive acquisition implements the first proposed free-choice policy. At each state it can acquire any remaining core laboratory feature or choose `STOP`; both the selected test and the stopping time can vary by retrospective participant.",
        "",
        "## Experiment Contract",
        "",
        f"- Cohort: `{results['dataset']['cohort_path']}`",
        f"- Cohort SHA-256: `{results['dataset']['cohort_sha256']}`",
        "- Predictor models and preprocessing: all 64 static baseline Cox subsets, fitted on train only.",
        "- Value models: six Ridge regressions fitted on train only.",
        "- Value target: absolute risk revision from acquiring one candidate; no event, duration, validation outcome, or test outcome is used directly.",
        "- Cost-penalty candidates: generated from train predicted benefit/cost only.",
        "- Cost-penalty selection: validation only; test evaluates one locked policy.",
        f"- Retention target: {100.0 * selected['retention_target']:.1f}% of validation C-index improvement from Base Only to Full Features.",
        f"- Primary cost scheme: `{results['costs']['default_scheme']}` relative units; no real prices are claimed.",
        "- No participant-level paths or predictions are written.",
        "",
        "## State And Actions",
        "",
        "Action space: "
        + ", ".join(f"`{action}`" for action in results["adaptive_policy"]["action_space"]),
        "",
        "The state contains transformed base values, acquired candidate values, acquisition masks, separate source-observation masks, current 3-year risk, remaining tests, and candidate costs. Unacquired candidate values, participant ID, site, event, duration, and follow-up fields are unavailable.",
        "",
        "For each remaining candidate, utility is `predicted absolute risk revision - lambda * cost`. The policy acquires the maximum-utility test when that utility is non-negative and otherwise chooses `STOP`.",
        "",
        "## Train-Only Value Models",
        "",
        "| Candidate | Training R2 | Mean target revision | Median target revision | P95 target revision |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in results["value_models"]["candidate_models"]:
        lines.append(
            f"| `{model['candidate']}` | {model['training_r2']:.4f} | "
            f"{model['target_mean_absolute_risk_revision']:.6f} | "
            f"{model['target_median_absolute_risk_revision']:.6f} | "
            f"{model['target_p95_absolute_risk_revision']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Validation Selection",
            "",
            f"Selected cost penalty: `{selected['cost_penalty']:.10f}`. It retains {100.0 * selected['retained_improvement_fraction']:.2f}% of the validation C-index improvement.",
            "",
            "| Policy point | Cost penalty | Mean tests | Validation C-index | Brier @ 3y | Stop at base | Acquire all | Unique paths |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, point in (
        ("Always STOP", results["tuning"]["always_stop_endpoint"]),
        ("Selected", selected),
        ("Always acquire", results["tuning"]["always_acquire_endpoint"]),
    ):
        penalty = "infinity" if point["cost_penalty"] is None else f"{point['cost_penalty']:.10f}"
        c_index = point.get("selected_validation_c_index", point.get("validation_c_index"))
        brier = point.get("selected_validation_ipcw_brier", point.get("validation_ipcw_brier"))
        lines.append(
            f"| {label} | {penalty} | {point['mean_acquired_tests']:.3f} | "
            f"{_metric(c_index)} | {_metric(brier)} | "
            f"{_percent(point['stop_without_labs_percent'])} | "
            f"{_percent(point['acquire_all_percent'])} | {point['unique_paths']} |"
        )

    lines.extend(
        [
            "",
            "## Predictive Performance",
            "",
            "| Method | Split | C-index | IPCW Brier @ 3y | Mean predicted risk | KM observed risk |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for label, split_name, metrics in (
        ("Base Only", "validation", base["validation"]),
        ("Full Features", "validation", full["validation"]),
        ("Ordered NEXT/STOP", "validation", ordered["validation"]["prediction"]),
        ("Adaptive", "validation", adaptive_validation["prediction"]),
        ("Base Only", "test", base["test"]),
        ("Full Features", "test", full["test"]),
        ("Ordered NEXT/STOP", "test", ordered["test"]["prediction"]),
        ("Adaptive", "test", adaptive_test["prediction"]),
    ):
        lines.append(
            f"| {label} | {split_name} | {_metric(metrics['harrell_c_index'])} | "
            f"{_metric(metrics['ipcw_brier_at_horizon'])} | "
            f"{_metric(metrics['mean_predicted_risk_at_horizon'])} | "
            f"{_metric(metrics['km_observed_risk_at_horizon'])} |"
        )

    lines.extend(
        [
            "",
            "## Acquisition Comparison",
            "",
            "| Method | Split | Mean tests/cost | Median tests | Stop at base | Acquire all | Unique paths | Final subsets |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, split_name, acquisition in (
        ("Ordered", "validation", ordered["validation"]["acquisition"]),
        ("Adaptive", "validation", adaptive_validation["acquisition"]),
        ("Ordered", "test", ordered["test"]["acquisition"]),
        ("Adaptive", "test", adaptive_test["acquisition"]),
    ):
        unique_paths = (
            ordered_path_counts[split_name]
            if label == "Ordered"
            else acquisition["unique_paths"]
        )
        final_subsets = (
            ordered_path_counts[split_name]
            if label == "Ordered"
            else acquisition["unique_final_subsets"]
        )
        lines.append(
            f"| {label} | {split_name} | "
            f"{acquisition['cost_by_scheme']['uniform_unit']['mean']:.3f} | "
            f"{acquisition['median_acquired_tests']:.1f} | "
            f"{_percent(acquisition['stop_without_labs_percent'])} | "
            f"{_percent(acquisition['acquire_all_percent'])} | "
            f"{unique_paths} | {final_subsets} |"
        )

    lines.extend(
        [
            "",
            "## Test Action Patterns",
            "",
            "### First Action",
            "",
            "| Action | Participants | Percent |",
            "|---|---:|---:|",
        ]
    )
    for row in test_acquisition["action_by_step"][0]["actions"]:
        lines.append(
            f"| `{row['action']}` | {row['participants']:,} | "
            f"{_percent(row['percent_of_all_participants'])} |"
        )

    lines.extend(
        [
            "",
            "### Acquired Features",
            "",
            "| Feature | Participants | Percent | Missing source results |",
            "|---|---:|---:|---:|",
        ]
    )
    missing_by_feature = {
        row["feature"]: row
        for row in test_acquisition["attempted_source_missingness"]
    }
    for row in test_acquisition["acquired_by_feature"]:
        missing = missing_by_feature[row["feature"]]
        lines.append(
            f"| `{row['feature']}` | {row['participants']:,} | "
            f"{_percent(row['percent'])} | {missing['source_value_missing']:,} |"
        )

    lines.extend(
        [
            "",
            "### Acquisition Count",
            "",
            "| Tests | Participants | Percent |",
            "|---:|---:|---:|",
        ]
    )
    for row in test_acquisition["count_distribution"]:
        lines.append(
            f"| {row['acquired_count']} | {row['participants']:,} | "
            f"{_percent(row['percent'])} |"
        )

    lines.extend(
        [
            "",
            "### Most Common Paths",
            "",
            "| Path | Participants | Percent |",
            "|---|---:|---:|",
        ]
    )
    for row in test_acquisition["top_paths"][:10]:
        lines.append(
            f"| {' -> '.join(row['path'])} | {row['participants']:,} | "
            f"{_percent(row['percent'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"On test, adaptive acquisition uses {adaptive_test_cost:.3f} relative units on average, a {vs_full_reduction:.1f}% reduction from Full Features and a {vs_ordered_reduction:.1f}% reduction from the ordered ordered acquisition policy. Its test C-index is {_metric(adaptive_test['prediction']['harrell_c_index'])}, compared with {_metric(ordered['test']['prediction']['harrell_c_index'])} ordered and {_metric(full['test']['harrell_c_index'])} full. The test set contains {test_acquisition['unique_paths']} distinct adaptive paths and {test_acquisition['unique_final_subsets']} final subsets.",
            "",
            "## Limitations",
            "",
            "- Predicted absolute risk revision is a pragmatic value-of-information surrogate, not a causal test benefit or guaranteed error reduction.",
            "- Linear Ridge value models may underfit nonlinear state-dependent utility.",
            "- Each candidate value model uses one randomly sampled train state per participant rather than every possible state.",
            "- Current risks come from separate subset-specific Cox models, not one mask-aware predictor.",
            "- Historical missingness cannot reveal counterfactual unrecorded laboratory values; charged imputation remains a simulation convention.",
            "- The same validation split selects the cost penalty and previously informed static model development.",
            "- Costs are relative feature-level units, with no shared-panel or real-price interpretation.",
            "- No uncertainty intervals, external validation, or site sensitivity analysis are included.",
            "",
        ]
    )
    return "\n".join(lines)


def run_adaptive_acquisition(
    *,
    project_root: Path,
    cohort_path: Path,
    cohort_manifest_path: Path,
    feature_config_path: Path,
    cost_config_path: Path,
    static_metrics_path: Path,
    static_model_path: Path,
    ordered_metrics_path: Path,
    adaptive_model_path: Path,
    metrics_path: Path,
    report_path: Path,
    retention_target: float = DEFAULT_RETENTION_TARGET,
    lambda_quantiles: int = DEFAULT_LAMBDA_QUANTILES,
    train_state_seed: int = TRAIN_STATE_SEED,
    ridge_alpha: float = VALUE_MODEL_ALPHA,
) -> dict[str, Any]:
    design = load_feature_cost_design(feature_config_path, cost_config_path)
    splits = load_cohort_splits(cohort_path)
    bundle = load_adaptive_model_bundle(static_metrics_path, static_model_path)

    print("Preparing all train and validation subset risks...", flush=True)
    train_data = prepare_adaptive_data(bundle, splits.train)
    validation_data = prepare_adaptive_data(bundle, splits.validation)
    print("Fitting six train-only candidate value models...", flush=True)
    value_models = fit_adaptive_value_models(
        train_data,
        design.costs,
        seed=train_state_seed,
        alpha=ridge_alpha,
        lambda_quantiles=lambda_quantiles,
    )
    print(
        f"Evaluating {len(value_models.lambda_candidates)} cost penalties on validation...",
        flush=True,
    )
    search_points = evaluate_lambda_grid(
        value_models.lambda_candidates,
        validation_data,
        splits.validation,
        splits.train,
        value_models,
        design.costs,
    )
    selected = select_adaptive_operating_point(
        search_points,
        retention_target=retention_target,
    )

    ordered_results = json.loads(ordered_metrics_path.read_text(encoding="utf-8"))
    _validate_provenance(
        bundle.static_results,
        ordered_results,
        cohort_path,
        selected,
    )
    validation_simulation = simulate_adaptive_policy(
        validation_data,
        value_models,
        design.costs,
        selected.point.cost_penalty,
    )
    print("Cost penalty locked; evaluating one adaptive policy on test...", flush=True)
    test_data = prepare_adaptive_data(bundle, splits.test)
    test_simulation = simulate_adaptive_policy(
        test_data,
        value_models,
        design.costs,
        selected.point.cost_penalty,
    )
    split_results = {
        "validation": {
            "prediction": adaptive_prediction_metrics(
                validation_simulation,
                splits.validation,
                splits.train,
            ),
            "acquisition": adaptive_acquisition_summary(
                validation_simulation,
                splits.validation,
                design.costs,
            ),
        },
        "test": {
            "prediction": adaptive_prediction_metrics(
                test_simulation,
                splits.test,
                splits.train,
            ),
            "acquisition": adaptive_acquisition_summary(
                test_simulation,
                splits.test,
                design.costs,
            ),
        },
    }
    save_adaptive_value_artifact(value_models, selected, adaptive_model_path)
    results = build_adaptive_results(
        project_root=project_root,
        cohort_path=cohort_path,
        cohort_manifest_path=cohort_manifest_path,
        feature_config_path=feature_config_path,
        cost_config_path=cost_config_path,
        static_metrics_path=static_metrics_path,
        static_model_path=static_model_path,
        ordered_metrics_path=ordered_metrics_path,
        adaptive_model_path=adaptive_model_path,
        static_results=bundle.static_results,
        ordered_results=ordered_results,
        value_models=value_models,
        search_points=search_points,
        selected=selected,
        lambda_quantiles=lambda_quantiles,
        split_results=split_results,
    )
    write_results_json(results, metrics_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_markdown_report(results), encoding="utf-8")
    return results
