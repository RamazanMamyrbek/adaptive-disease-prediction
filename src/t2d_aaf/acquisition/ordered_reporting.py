from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

from t2d_aaf.acquisition.ordered import (
    ACTION_NAMES,
    DEFAULT_RETENTION_TARGET,
    DEFAULT_THRESHOLD_QUANTILES,
    ORDER_SOURCE_METHOD,
    POLICY_NAME,
    SelectedOperatingPoint,
    ThresholdSearchPoint,
    acquisition_summary,
    build_train_threshold_candidates,
    evaluate_threshold_grid,
    load_ordered_model_bundle,
    prediction_metrics,
    predict_ordered_stage_risks,
    select_operating_point,
    simulate_ordered_policy,
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


def _threshold_json(value: float) -> float | None:
    return None if math.isinf(value) else float(value)


def _search_point_json(point: ThresholdSearchPoint) -> dict[str, float | None]:
    output = asdict(point)
    output["threshold"] = _threshold_json(point.threshold)
    return output


def _selected_point_json(selected: SelectedOperatingPoint) -> dict[str, Any]:
    return {
        "threshold": _threshold_json(selected.point.threshold),
        "retention_target": selected.retention_target,
        "target_validation_c_index": selected.target_validation_c_index,
        "base_validation_c_index": selected.base_validation_c_index,
        "full_validation_c_index": selected.full_validation_c_index,
        "selected_validation_c_index": selected.point.validation_c_index,
        "selected_validation_ipcw_brier": selected.point.validation_ipcw_brier,
        "retained_improvement_fraction": selected.retained_improvement_fraction,
        "mean_acquired_tests": selected.point.mean_acquired_tests,
        "median_acquired_tests": selected.point.median_acquired_tests,
        "stop_without_labs_percent": selected.point.stop_without_labs_percent,
        "acquire_all_percent": selected.point.acquire_all_percent,
    }


def _validate_static_provenance(
    static_results: Mapping[str, Any],
    cohort_path: Path,
    selected: SelectedOperatingPoint,
) -> None:
    actual_cohort_hash = sha256_file(cohort_path)
    if static_results["dataset"]["cohort_sha256"] != actual_cohort_hash:
        raise ValueError("static baseline metrics reference a different cohort file.")
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
        raise ValueError("Ordered always-STOP endpoint differs from static baseline Base Only.")
    if not math.isclose(
        selected.full_validation_c_index, static_full, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Ordered always-NEXT endpoint differs from static baseline Full Features.")


def build_ordered_results(
    *,
    project_root: Path,
    cohort_path: Path,
    cohort_manifest_path: Path,
    feature_config_path: Path,
    cost_config_path: Path,
    static_metrics_path: Path,
    static_model_path: Path,
    order: tuple[str, ...],
    static_results: Mapping[str, Any],
    search_points: tuple[ThresholdSearchPoint, ...],
    selected: SelectedOperatingPoint,
    quantile_count: int,
    split_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    design = load_feature_cost_design(feature_config_path, cost_config_path)
    always_next = next(point for point in search_points if point.threshold == 0.0)
    always_stop = next(point for point in search_points if math.isinf(point.threshold))
    return {
        "schema_version": "ordered_acquisition_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 6,
        "scope": "fixed global order with patient-specific NEXT/STOP; no free-choice acquisition",
        "seeds": {"cohort_split": DEFAULT_SEED, "policy_randomness": None},
        "dataset": {
            "cohort_path": portable_path(cohort_path, project_root),
            "cohort_sha256": sha256_file(cohort_path),
            "cohort_manifest_path": portable_path(cohort_manifest_path, project_root),
            "cohort_manifest_sha256": sha256_file(cohort_manifest_path),
        },
        "static_provenance": {
            "metrics_path": portable_path(static_metrics_path, project_root),
            "metrics_sha256": sha256_file(static_metrics_path),
            "local_model_path": portable_path(static_model_path, project_root),
            "local_model_sha256": sha256_file(static_model_path),
            "models_fit_split": static_results["model"]["fit_split"],
            "preprocessing_fit_split": static_results["preprocessing"]["fit_split"],
            "model_type": static_results["model"]["type"],
            "ties": static_results["model"]["ties"],
            "optimizer": static_results["model"]["optimizer"],
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
        "fixed_order": {
            "source_stage": 5,
            "source_method": ORDER_SOURCE_METHOD,
            "selection_data": "validation only in static baseline",
            "order": list(order),
            "patient_specific_order": False,
        },
        "state_contract": {
            "observed_values": "base features plus the acquired fixed-order prefix",
            "acquisition_mask": "base observed by contract; candidate prefix marked acquired",
            "source_observation_mask": (
                "separately records whether each retrospectively requested source value exists"
            ),
            "current_prediction": "3-year incident-T2D risk from the current prefix Cox model",
            "available_next_action": "the next single candidate in the fixed order",
            "candidate_cost": "cost of that next candidate under the named scheme",
            "outcome_or_followup_available_to_policy": False,
            "identifier_or_site_available_to_policy": False,
        },
        "ordered_policy": {
            "name": POLICY_NAME,
            "action_space": list(ACTION_NAMES),
            "free_choice_among_remaining_tests": False,
            "decision_rule": (
                "at stages 0-5 choose NEXT when current 3-year risk >= threshold; "
                "otherwise STOP; force STOP after stage 6"
            ),
            "threshold_shared_across_stages": True,
            "selected_operating_point": _selected_point_json(selected),
            "missing_result_rule": (
                "charge the requested action, retain the participant, and use the "
                "train-fitted imputation value when the retrospective source result is missing"
            ),
        },
        "tuning": {
            "threshold_candidate_source": "quantiles of train prefix risks at decision stages",
            "threshold_quantile_count_requested": quantile_count,
            "threshold_candidates_evaluated": len(search_points),
            "selection_split": "validation",
            "selection_metric": "Harrell C-index retention relative to Base Only and Full Features",
            "selection_rule": (
                "minimum mean primary-scheme cost among candidates retaining the target "
                "fraction of the validation C-index improvement"
            ),
            "test_used_for_threshold_selection": False,
            "test_thresholds_evaluated": 1,
            "always_stop_endpoint": _search_point_json(always_stop),
            "always_next_endpoint": _search_point_json(always_next),
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
        "adaptive_acquisition_policy": None,
        "environment": package_versions(),
    }


def write_results_json(results: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def _metric(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _percent(value: float) -> str:
    return f"{value:.2f}%"


def build_markdown_report(results: Mapping[str, Any]) -> str:
    policy = results["ordered_policy"]
    selected = policy["selected_operating_point"]
    evaluation = results["evaluation"]
    validation = evaluation["selected_policy"]["validation"]
    test = evaluation["selected_policy"]["test"]
    base = evaluation["static_references"]["base_only"]
    full = evaluation["static_references"]["full_features"]
    order = results["fixed_order"]["order"]
    test_acquisition = test["acquisition"]
    cost_reduction = 100.0 * (
        1.0
        - test_acquisition["cost_by_scheme"]["uniform_unit"]["mean"]
        / len(order)
    )
    lines = [
        "# ordered acquisition Ordered NEXT/STOP Baseline",
        "",
        "ordered acquisition implements a SAGEAgent-like ordered acquisition baseline: the laboratory order is fixed globally, while each retrospective participant receives a patient-specific `NEXT` or `STOP` decision. It does not allow free choice among remaining tests and therefore is not the proposed adaptive method from adaptive acquisition.",
        "",
        "## Experiment Contract",
        "",
        f"- Cohort: `{results['dataset']['cohort_path']}`",
        f"- Cohort SHA-256: `{results['dataset']['cohort_sha256']}`",
        "- Reused predictors: static baseline Cox prefix models and preprocessing fitted on train only.",
        "- Fixed order source: static baseline validation-greedy order; test did not select the order.",
        "- Threshold candidates: train-risk quantiles only.",
        "- Threshold selection: validation only; test evaluates one locked threshold.",
        f"- Retention target: {100.0 * selected['retention_target']:.1f}% of the validation C-index improvement from Base Only to Full Features.",
        f"- Primary cost scheme: `{results['costs']['default_scheme']}` relative units; no real prices are claimed.",
        "- No participant-level paths or predictions are written.",
        "",
        "## Fixed Order",
        "",
        " -> ".join(f"`{feature}`" for feature in order),
        "",
        "The only actions at stages 0-5 are `NEXT` (acquire the next feature above) and `STOP`. After all six candidates, `STOP` is forced. The next feature itself is never patient-specific in this stage.",
        "",
        "## Policy State And Rule",
        "",
        "The state contains base values, acquired-prefix values, an acquisition mask, a separate source-observation mask, the current 3-year risk, the one available next test, and its cost. It never contains participant ID, site, event, duration, or follow-up leakage fields.",
        "",
        f"Selected threshold: `{selected['threshold']:.8f}`. At every decision stage, choose `NEXT` when current predicted 3-year risk is at least this threshold; otherwise choose `STOP`.",
        "",
        "## Validation Selection",
        "",
        "| Policy point | Threshold | Mean tests | Validation C-index | Validation Brier @ 3y | Stop at base | Acquire all |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, point in (
        ("Always STOP", results["tuning"]["always_stop_endpoint"]),
        ("Selected", selected),
        ("Always NEXT", results["tuning"]["always_next_endpoint"]),
    ):
        threshold = "infinity" if point["threshold"] is None else f"{point['threshold']:.8f}"
        c_index = point.get("selected_validation_c_index", point.get("validation_c_index"))
        brier = point.get("selected_validation_ipcw_brier", point.get("validation_ipcw_brier"))
        lines.append(
            f"| {label} | {threshold} | {point['mean_acquired_tests']:.3f} | "
            f"{_metric(c_index)} | {_metric(brier)} | "
            f"{_percent(point['stop_without_labs_percent'])} | "
            f"{_percent(point['acquire_all_percent'])} |"
        )

    lines.extend(
        [
            "",
            "## Predictive Performance",
            "",
            "static baseline endpoints are shown as inherited static references. Only the selected ordered threshold is newly evaluated on test in ordered acquisition.",
            "",
            "| Method | Split | C-index | IPCW Brier @ 3y | Mean predicted risk | KM observed risk |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for label, split_name, metrics in (
        ("Base Only reference", "validation", base["validation"]),
        ("Full Features reference", "validation", full["validation"]),
        ("Ordered NEXT/STOP", "validation", validation["prediction"]),
        ("Base Only reference", "test", base["test"]),
        ("Full Features reference", "test", full["test"]),
        ("Ordered NEXT/STOP", "test", test["prediction"]),
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
            "## Acquisition Outcomes",
            "",
            "| Split | N | Mean tests | Median tests | Mean primary cost | Stop at base | Acquire all |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split_name, split_result in (("validation", validation), ("test", test)):
        acquisition = split_result["acquisition"]
        lines.append(
            f"| {split_name} | {split_result['prediction']['participants']:,} | "
            f"{acquisition['mean_acquired_tests']:.3f} | "
            f"{acquisition['median_acquired_tests']:.1f} | "
            f"{acquisition['cost_by_scheme']['uniform_unit']['mean']:.3f} | "
            f"{_percent(acquisition['stop_without_labs_percent'])} | "
            f"{_percent(acquisition['acquire_all_percent'])} |"
        )

    lines.extend(
        [
            "",
            "### Test Stop Distribution",
            "",
            "| Acquired tests | Acquired prefix | Participants | Percent |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in test_acquisition["stage_distribution"]:
        prefix = ", ".join(row["acquired_features"]) or "base only"
        lines.append(
            f"| {row['acquired_count']} | {prefix} | {row['participants']:,} | "
            f"{_percent(row['percent'])} |"
        )

    lines.extend(
        [
            "",
            "## Retrospective Missing Results",
            "",
            "A requested action is charged even when its historical source value is missing. The corresponding static baseline train-fitted imputation value is used, the participant remains in the simulation, and the policy continues normally.",
            "",
            "| Split | Feature | Attempts | Missing source values | Missing among attempts |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for split_name, split_result in (("validation", validation), ("test", test)):
        for row in split_result["acquisition"]["attempted_source_missingness"]:
            lines.append(
                f"| {split_name} | `{row['feature']}` | {row['attempted']:,} | "
                f"{row['source_value_missing']:,} | "
                f"{_percent(row['missing_percent_of_attempts'])} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"On test, the selected ordered policy uses {test_acquisition['mean_acquired_tests']:.3f} laboratory actions on average, a {cost_reduction:.1f}% reduction from always acquiring all six under `uniform_unit`. Its test C-index is {_metric(test['prediction']['harrell_c_index'])}, compared with {_metric(full['test']['harrell_c_index'])} for the inherited Full Features reference. This is evidence about one retrospective hold-out split, not a clinical claim.",
            "",
            "## Limitations",
            "",
            "- The fixed order and STOP threshold are both developed on the same single validation split.",
            "- The one-sided risk gate is deliberately simple and may continue all tests for many higher-risk participants.",
            "- Prefix predictions come from separate Cox models rather than one mask-aware predictor.",
            "- Historical missingness cannot reveal the counterfactual value of a laboratory test that was not recorded; charged imputation is a conservative simulation convention.",
            "- Costs are relative feature-level units and do not represent prices, shared blood draws, panels, or patient burden.",
            "- No uncertainty intervals, site sensitivity analysis, or external validation are included.",
            "- The next test is globally fixed; free-choice patient-specific acquisition remains adaptive acquisition.",
            "",
        ]
    )
    return "\n".join(lines)


def run_ordered_acquisition(
    *,
    project_root: Path,
    cohort_path: Path,
    cohort_manifest_path: Path,
    feature_config_path: Path,
    cost_config_path: Path,
    static_metrics_path: Path,
    static_model_path: Path,
    metrics_path: Path,
    report_path: Path,
    retention_target: float = DEFAULT_RETENTION_TARGET,
    threshold_quantiles: int = DEFAULT_THRESHOLD_QUANTILES,
) -> dict[str, Any]:
    design = load_feature_cost_design(feature_config_path, cost_config_path)
    splits = load_cohort_splits(cohort_path)
    bundle = load_ordered_model_bundle(static_metrics_path, static_model_path)

    print("Predicting ordered stages for train and validation...", flush=True)
    train_stage_risks = predict_ordered_stage_risks(bundle, splits.train)
    validation_stage_risks = predict_ordered_stage_risks(bundle, splits.validation)
    thresholds = build_train_threshold_candidates(
        train_stage_risks,
        quantile_count=threshold_quantiles,
    )
    print(f"Evaluating {len(thresholds)} thresholds on validation...", flush=True)
    search_points = evaluate_threshold_grid(
        thresholds,
        validation_stage_risks,
        splits.validation,
        splits.train,
        design.costs,
        bundle.order,
    )
    selected = select_operating_point(search_points, retention_target=retention_target)
    _validate_static_provenance(bundle.static_results, cohort_path, selected)

    validation_simulation = simulate_ordered_policy(
        validation_stage_risks,
        selected.point.threshold,
    )
    print("Threshold locked; evaluating one policy on test...", flush=True)
    test_stage_risks = predict_ordered_stage_risks(bundle, splits.test)
    test_simulation = simulate_ordered_policy(test_stage_risks, selected.point.threshold)
    split_results = {
        "validation": {
            "prediction": prediction_metrics(
                validation_simulation,
                splits.validation,
                splits.train,
            ),
            "acquisition": acquisition_summary(
                validation_simulation,
                splits.validation,
                design.costs,
                bundle.order,
            ),
        },
        "test": {
            "prediction": prediction_metrics(test_simulation, splits.test, splits.train),
            "acquisition": acquisition_summary(
                test_simulation,
                splits.test,
                design.costs,
                bundle.order,
            ),
        },
    }
    results = build_ordered_results(
        project_root=project_root,
        cohort_path=cohort_path,
        cohort_manifest_path=cohort_manifest_path,
        feature_config_path=feature_config_path,
        cost_config_path=cost_config_path,
        static_metrics_path=static_metrics_path,
        static_model_path=static_model_path,
        order=bundle.order,
        static_results=bundle.static_results,
        search_points=search_points,
        selected=selected,
        quantile_count=threshold_quantiles,
        split_results=split_results,
    )
    write_results_json(results, metrics_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_markdown_report(results), encoding="utf-8")
    return results
