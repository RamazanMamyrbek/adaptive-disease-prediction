from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from numpy.typing import NDArray

from t2d_aaf.acquisition.adaptive import (
    DEFAULT_LAMBDA_QUANTILES,
    FittedAdaptiveValueModels,
    build_cost_penalty_candidates,
    load_adaptive_model_bundle,
    prepare_adaptive_data,
    simulate_adaptive_policy,
    subset_to_mask,
)
from t2d_aaf.acquisition.ordered import (
    prefix_costs,
    simulate_ordered_policy,
)
from t2d_aaf.data.cohort import (
    BASE_FEATURES,
    CORE_ACQUIRABLE_FEATURES,
    DEFAULT_SEED,
    EXCLUDED_FROM_MAIN_PREDICTORS,
    OPTIONAL_FEATURES,
)
from t2d_aaf.evaluation.cost_performance import (
    paired_bootstrap_evaluation,
    prediction_evaluation,
)
from t2d_aaf.evaluation.survival_metrics import harrell_c_index, ipcw_brier_score
from t2d_aaf.features.registry import CostConfig, load_feature_cost_design
from t2d_aaf.survival.baseline import EVALUATION_HORIZON_YEARS, load_cohort_splits
from t2d_aaf.survival.reporting import package_versions, portable_path, sha256_file


EVALUATION_SCHEMA_VERSION = "cost_performance_evaluation_v1"
EVALUATION_BOOTSTRAP_SEED = 20260827
DEFAULT_BOOTSTRAP_REPLICATES = 200
CALIBRATION_BIN_COUNT = 10
PRIMARY_RETENTION_TARGET = 0.99

METHOD_LABELS: dict[str, str] = {
    "base_only": "Base Only",
    "full_features": "Full Features",
    "static_top_k": "Static Top-K",
    "random_acquisition": "Random Acquisition",
    "importance_greedy": "Importance Greedy",
    "ordered_acquisition": "Ordered NEXT/STOP",
    "adaptive_acquisition": "Proposed Adaptive",
}


def _read_json(path: Path, expected_stage: int) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("stage") != expected_stage:
        raise ValueError(f"Expected Stage {expected_stage} metrics at {path}.")
    return document


def _load_value_models(path: Path) -> FittedAdaptiveValueModels:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != (
        "adaptive_value_policy_v1"
    ):
        raise ValueError("Unsupported or invalid adaptive acquisition value-model artifact.")
    if tuple(artifact.get("candidate_features", ())) != CORE_ACQUIRABLE_FEATURES:
        raise ValueError("adaptive acquisition artifact candidate order differs from the cohort contract.")
    models = tuple(artifact.get("value_models", ()))
    if len(models) != len(CORE_ACQUIRABLE_FEATURES):
        raise ValueError("adaptive acquisition artifact must contain one value model per candidate.")
    return FittedAdaptiveValueModels(
        models=models,
        state_feature_names=tuple(artifact["state_feature_names"]),
        lambda_candidates=np.empty(0, dtype=float),
        seed=int(artifact["training_seed"]),
        alpha=float(artifact["ridge_alpha"]),
    )


def _mask_costs(costs: CostConfig, scheme: str) -> NDArray[np.float64]:
    values = []
    for mask in range(2 ** len(CORE_ACQUIRABLE_FEATURES)):
        subset = tuple(
            feature
            for index, feature in enumerate(CORE_ACQUIRABLE_FEATURES)
            if mask & (1 << index)
        )
        values.append(costs.acquisition_cost(subset, scheme=scheme))
    return np.asarray(values, dtype=float)


def _acquisition_metrics(
    acquired_count: NDArray[np.int64],
    participant_cost: NDArray[np.float64],
) -> dict[str, float]:
    if len(acquired_count) != len(participant_cost):
        raise ValueError("Acquisition-count and cost vectors must have equal lengths.")
    candidate_count = len(CORE_ACQUIRABLE_FEATURES)
    return {
        "mean_acquired_tests": float(np.mean(acquired_count)),
        "median_acquired_tests": float(np.median(acquired_count)),
        "p25_acquired_tests": float(np.quantile(acquired_count, 0.25)),
        "p75_acquired_tests": float(np.quantile(acquired_count, 0.75)),
        "mean_cost": float(np.mean(participant_cost)),
        "median_cost": float(np.median(participant_cost)),
        "p25_cost": float(np.quantile(participant_cost, 0.25)),
        "p75_cost": float(np.quantile(participant_cost, 0.75)),
        "stop_without_labs_percent": float(100.0 * np.mean(acquired_count == 0)),
        "acquire_all_percent": float(100.0 * np.mean(acquired_count == candidate_count)),
    }


def _fixed_acquisition_metrics(
    participants: int,
    acquired_count: int,
    cost: float,
) -> tuple[dict[str, float], NDArray[np.float64], NDArray[np.int64]]:
    counts = np.full(participants, acquired_count, dtype=np.int64)
    costs = np.full(participants, cost, dtype=float)
    return _acquisition_metrics(counts, costs), costs, counts


def _prediction_metrics(
    train_frame: Any,
    evaluation_frame: Any,
    risk: NDArray[np.float64],
) -> dict[str, object]:
    return prediction_evaluation(
        train_frame["duration_years"].to_numpy(dtype=float),
        train_frame["event"].to_numpy(dtype=np.int64),
        evaluation_frame["duration_years"].to_numpy(dtype=float),
        evaluation_frame["event"].to_numpy(dtype=np.int64),
        risk,
        EVALUATION_HORIZON_YEARS,
        calibration_bin_count=CALIBRATION_BIN_COUNT,
    )


def _validation_target(
    static_results: Mapping[str, Any], retention_target: float
) -> dict[str, float]:
    base = float(
        static_results["methods"]["base_only"]["result"]["metrics"]["validation"][
            "harrell_c_index"
        ]
    )
    full = float(
        static_results["methods"]["full_features"]["result"]["metrics"]["validation"][
            "harrell_c_index"
        ]
    )
    if full <= base:
        raise ValueError("Full Features must improve validation C-index over Base Only.")
    return {
        "retention_fraction": retention_target,
        "base_validation_c_index": base,
        "full_validation_c_index": full,
        "target_validation_c_index": base + retention_target * (full - base),
    }


def _select_static_point(
    static_results: Mapping[str, Any], method: str, target: float
) -> Mapping[str, Any]:
    points = static_results["methods"][method]["points"]
    feasible = [
        point
        for point in points
        if float(point["metrics"]["validation"]["harrell_c_index"]) >= target
    ]
    if not feasible:
        raise ValueError(f"No {method} point satisfies the validation retention target.")
    return min(
        feasible,
        key=lambda point: (
            int(point["acquired_count"]),
            -float(point["metrics"]["validation"]["harrell_c_index"]),
        ),
    )


def _select_random_point(
    static_results: Mapping[str, Any], target: float
) -> Mapping[str, Any]:
    points = static_results["methods"]["random"]["points"]
    feasible = [
        point
        for point in points
        if float(point["metrics"]["validation"]["harrell_c_index"]["mean"])
        >= target
    ]
    if not feasible:
        raise ValueError("No Random Acquisition point satisfies the validation target.")
    return min(feasible, key=lambda point: int(point["acquired_count"]))


def _parameter(value: Any) -> float:
    return math.inf if value is None else float(value)


def _json_parameter(value: float) -> float | None:
    return None if math.isinf(value) else float(value)


def _select_tradeoff_point(
    points: Sequence[Mapping[str, Any]], target: float
) -> Mapping[str, Any]:
    feasible = [
        point for point in points if float(point["validation_c_index"]) >= target
    ]
    if not feasible:
        raise ValueError("No sensitivity operating point satisfies the validation target.")
    return min(
        feasible,
        key=lambda point: (
            float(point["mean_cost"]),
            -float(point["validation_c_index"]),
            float(point["validation_ipcw_brier"]),
            -_parameter(point["parameter"]),
        ),
    )


def _ordered_scheme_evaluation(
    *,
    scheme: str,
    ordered_results: Mapping[str, Any],
    validation_stage_risks: NDArray[np.float64],
    test_stage_risks: NDArray[np.float64],
    test_frame: Any,
    train_frame: Any,
    costs: CostConfig,
    order: tuple[str, ...],
    target: float,
) -> dict[str, Any]:
    validation_prefix_cost = prefix_costs(costs, order, scheme=scheme)
    points: list[dict[str, Any]] = []
    for source in ordered_results["tuning"]["search"]:
        threshold = _parameter(source["threshold"])
        simulation = simulate_ordered_policy(validation_stage_risks, threshold)
        points.append(
            {
                "parameter": _json_parameter(threshold),
                "mean_cost": float(
                    np.mean(validation_prefix_cost[simulation.stop_stage])
                ),
                "mean_acquired_tests": float(np.mean(simulation.stop_stage)),
                "validation_c_index": float(source["validation_c_index"]),
                "validation_ipcw_brier": float(source["validation_ipcw_brier"]),
            }
        )
    selected = _select_tradeoff_point(points, target)
    threshold = _parameter(selected["parameter"])
    test_simulation = simulate_ordered_policy(test_stage_risks, threshold)
    test_cost = prefix_costs(costs, order, scheme=scheme)[test_simulation.stop_stage]
    return {
        "selection_split": "validation",
        "test_parameters_evaluated": 1,
        "selected": dict(selected),
        "test": {
            "prediction": _prediction_metrics(
                train_frame, test_frame, test_simulation.selected_risk
            ),
            "acquisition": _acquisition_metrics(
                test_simulation.stop_stage.astype(np.int64), test_cost
            ),
        },
        "validation_curve": points,
    }


def _adaptive_scheme_evaluation(
    *,
    scheme: str,
    train_data: Any,
    validation_data: Any,
    test_data: Any,
    validation_frame: Any,
    test_frame: Any,
    train_frame: Any,
    value_models: FittedAdaptiveValueModels,
    costs: CostConfig,
    target: float,
    lambda_quantiles: int,
) -> dict[str, Any]:
    penalties = build_cost_penalty_candidates(
        train_data,
        value_models,
        costs,
        scheme=scheme,
        lambda_quantiles=lambda_quantiles,
    )
    mask_cost = _mask_costs(costs, scheme)
    duration = validation_frame["duration_years"].to_numpy(dtype=float)
    event = validation_frame["event"].to_numpy(dtype=np.int64)
    train_duration = train_frame["duration_years"].to_numpy(dtype=float)
    train_event = train_frame["event"].to_numpy(dtype=np.int64)
    points: list[dict[str, Any]] = []
    for penalty in penalties:
        simulation = simulate_adaptive_policy(
            validation_data,
            value_models,
            costs,
            float(penalty),
            scheme=scheme,
        )
        participant_cost = mask_cost[simulation.final_masks]
        points.append(
            {
                "parameter": _json_parameter(float(penalty)),
                "mean_cost": float(np.mean(participant_cost)),
                "mean_acquired_tests": float(np.mean(simulation.acquired_count)),
                "validation_c_index": harrell_c_index(
                    duration, event, simulation.selected_risk
                ),
                "validation_ipcw_brier": ipcw_brier_score(
                    train_duration,
                    train_event,
                    duration,
                    event,
                    1.0 - simulation.selected_risk,
                    EVALUATION_HORIZON_YEARS,
                ),
            }
        )
    selected = _select_tradeoff_point(points, target)
    penalty = _parameter(selected["parameter"])
    test_simulation = simulate_adaptive_policy(
        test_data,
        value_models,
        costs,
        penalty,
        scheme=scheme,
    )
    test_cost = mask_cost[test_simulation.final_masks]
    return {
        "selection_split": "validation",
        "penalty_candidates_from": "train predicted benefit divided by named-scheme cost",
        "penalties_evaluated": len(points),
        "test_parameters_evaluated": 1,
        "selected": dict(selected),
        "test": {
            "prediction": _prediction_metrics(
                train_frame, test_frame, test_simulation.selected_risk
            ),
            "acquisition": _acquisition_metrics(
                test_simulation.acquired_count, test_cost
            ),
        },
        "validation_curve": points,
    }


def _primary_curves(
    static_results: Mapping[str, Any],
    ordered_results: Mapping[str, Any],
    adaptive_results: Mapping[str, Any],
) -> dict[str, list[dict[str, float | int | None]]]:
    curves: dict[str, list[dict[str, float | int | None]]] = {}
    for method in ("static_top_k", "importance_greedy"):
        curves[method] = [
            {
                "mean_cost": float(point["acquired_count"]),
                "acquired_count": int(point["acquired_count"]),
                "validation_c_index": float(
                    point["metrics"]["validation"]["harrell_c_index"]
                ),
                "validation_ipcw_brier": float(
                    point["metrics"]["validation"]["ipcw_brier_at_horizon"]
                ),
            }
            for point in static_results["methods"][method]["points"]
        ]
    curves["random_acquisition"] = [
        {
            "mean_cost": float(point["acquired_count"]),
            "acquired_count": int(point["acquired_count"]),
            "validation_c_index": float(
                point["metrics"]["validation"]["harrell_c_index"]["mean"]
            ),
            "validation_ipcw_brier": float(
                point["metrics"]["validation"]["ipcw_brier_at_horizon"]["mean"]
            ),
        }
        for point in static_results["methods"]["random"]["points"]
    ]
    curves["ordered_acquisition"] = [
        {
            "mean_cost": float(point["mean_acquired_tests"]),
            "parameter": point["threshold"],
            "validation_c_index": float(point["validation_c_index"]),
            "validation_ipcw_brier": float(point["validation_ipcw_brier"]),
        }
        for point in ordered_results["tuning"]["search"]
    ]
    curves["adaptive_acquisition"] = [
        {
            "mean_cost": float(point["mean_primary_cost"]),
            "parameter": point["cost_penalty"],
            "validation_c_index": float(point["validation_c_index"]),
            "validation_ipcw_brier": float(point["validation_ipcw_brier"]),
        }
        for point in adaptive_results["tuning"]["search"]
    ]
    return curves


def _write_figures(
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    primary: Mapping[str, Mapping[str, Any]],
    target: float,
    cost_figure_path: Path,
    calibration_figure_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cost_figure_path.parent.mkdir(parents=True, exist_ok=True)
    styles = {
        "static_top_k": ("Static Top-K", "#25706f", "o"),
        "random_acquisition": ("Random", "#777777", "s"),
        "importance_greedy": ("Importance Greedy", "#d07a30", "^"),
        "ordered_acquisition": ("Ordered NEXT/STOP", "#4063a5", "D"),
        "adaptive_acquisition": ("Proposed Adaptive", "#a33f55", "P"),
    }
    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for method, (label, color, marker) in styles.items():
        points = sorted(curves[method], key=lambda point: float(point["mean_cost"]))
        axis.plot(
            [float(point["mean_cost"]) for point in points],
            [float(point["validation_c_index"]) for point in points],
            label=label,
            color=color,
            marker=marker,
            markersize=4,
            markevery=max(1, len(points) // 15),
            linewidth=1.5,
            alpha=0.9,
        )
    axis.axhline(target, color="#222222", linestyle="--", linewidth=1.0, label="99% target")
    axis.set_xlabel("Mean relative cost on validation")
    axis.set_ylabel("Validation Harrell C-index")
    axis.set_xlim(left=-0.1)
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(cost_figure_path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    calibration_methods = (
        "base_only",
        "full_features",
        "static_top_k",
        "ordered_acquisition",
        "adaptive_acquisition",
    )
    calibration_colors = ("#777777", "#222222", "#25706f", "#4063a5", "#a33f55")
    maximum = 0.0
    for method, color in zip(calibration_methods, calibration_colors, strict=True):
        bins = primary[method]["test"]["prediction"]["calibration_bins"]
        predicted = [float(row["mean_predicted_risk"]) for row in bins]
        observed = [float(row["km_observed_risk"]) for row in bins]
        maximum = max(maximum, *predicted, *observed)
        axis.plot(
            predicted,
            observed,
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.4,
            label=METHOD_LABELS[method],
        )
    limit = max(0.02, maximum * 1.06)
    axis.plot([0.0, limit], [0.0, limit], color="#111111", linestyle="--", linewidth=1.0)
    axis.set_xlim(0.0, limit)
    axis.set_ylim(0.0, limit)
    axis.set_xlabel("Mean predicted 3-year risk")
    axis.set_ylabel("Kaplan-Meier observed 3-year risk")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8)
    figure.savefig(calibration_figure_path, dpi=180)
    plt.close(figure)


def _format_metric(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _format_interval(interval: Mapping[str, Any], digits: int = 4) -> str:
    return (
        f"{float(interval['point_estimate']):.{digits}f} "
        f"[{float(interval['ci95_lower']):.{digits}f}, "
        f"{float(interval['ci95_upper']):.{digits}f}]"
    )


def render_evaluation_report(results: Mapping[str, Any]) -> str:
    primary = results["primary_comparison"]
    bootstrap = results["bootstrap"]
    bootstrap_alias = {
        "base_only": "base_only",
        "full_features": "full_features",
        "static_top_k": "static_top_k",
        "random_acquisition": "full_features",
        "importance_greedy": "static_top_k",
        "ordered_acquisition": "ordered_acquisition",
        "adaptive_acquisition": "adaptive_acquisition",
    }
    lines = [
        "# cost-performance evaluation - Cost-Performance Evaluation",
        "",
        "## Scope and lock",
        "",
        "cost-performance evaluation evaluates the already fitted baseline modelling-7 predictors and policies. It does not fit a new predictor, value model, or acquisition architecture. Every primary operating point is selected on validation using the same 99% Base-to-Full C-index-retention target; test is used only after that lock.",
        "",
        f"- Horizon: {results['evaluation']['horizon_years']:.1f} years.",
        f"- Bootstrap: {bootstrap['replicates_requested']} participant-level paired replicates, stratified by event, seed `{bootstrap['seed']}`.",
        "- Costs are relative experimental units, not prices.",
        "- No participant-level predictions, paths, or bootstrap samples are written.",
        "",
        "## Primary test comparison",
        "",
        "Values are point estimates with 95% percentile bootstrap intervals. Random at its selected K=6 is identical to Full Features; Top-K and Greedy select the same K=3 subset, so those pairs share intervals.",
        "",
        "| Method | Selected validation rule | C-index (95% CI) | IPCW Brier@3y (95% CI) | IPCW AUC@3y (95% CI) | IPCW PR-AUC@3y (95% CI) | Mean cost (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_LABELS:
        record = primary[method]
        intervals = bootstrap["methods"][bootstrap_alias[method]]
        lines.append(
            f"| {METHOD_LABELS[method]} | {record['selection_summary']} | "
            f"{_format_interval(intervals['harrell_c_index'])} | "
            f"{_format_interval(intervals['ipcw_brier_at_horizon'])} | "
            f"{_format_interval(intervals['ipcw_cumulative_dynamic_auc'])} | "
            f"{_format_interval(intervals['ipcw_average_precision'])} | "
            f"{_format_interval(intervals['mean_cost'], 3)} |"
        )

    lines.extend(
        [
            "",
            "![Validation cost-performance curves](figures/cost_performance_uniform.png)",
            "",
            "The adaptive policy has the lowest mean cost among methods meeting the pre-specified validation target. On test, its discrimination is close to the ordered policy but below the static K=3 and Full Features estimates; this distinction matters and is not hidden by the cost reduction.",
            "",
            "## Acquisition burden",
            "",
            "| Method | Mean tests | Median tests | STOP at base | Acquire all six | Mean cost |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHOD_LABELS:
        acquisition = primary[method]["test"]["acquisition"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {_format_metric(acquisition['mean_acquired_tests'], 3)} | "
            f"{_format_metric(acquisition['median_acquired_tests'], 1)} | "
            f"{_format_metric(acquisition['stop_without_labs_percent'], 2)}% | "
            f"{_format_metric(acquisition['acquire_all_percent'], 2)}% | "
            f"{_format_metric(acquisition['mean_cost'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Calibration",
            "",
            "Calibration uses ten equally sized risk-rank bins. Observed 3-year risk in each bin is estimated as 1 - Kaplan-Meier survival, so early censoring is not coded as a non-event.",
            "",
            "| Method | Mean predicted risk | KM observed risk | Calibration-in-large | Mean absolute bin gap | Maximum bin gap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHOD_LABELS:
        metrics = primary[method]["test"]["prediction"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {_format_metric(metrics['mean_predicted_risk_at_horizon'])} | "
            f"{_format_metric(metrics['km_observed_risk_at_horizon'])} | "
            f"{_format_metric(metrics['calibration_in_the_large_difference'])} | "
            f"{_format_metric(metrics['calibration_mean_absolute_bin_gap'])} | "
            f"{_format_metric(metrics['calibration_maximum_absolute_bin_gap'])} |"
        )
    lines.extend(
        [
            "",
            "![Test calibration](figures/calibration_test.png)",
            "",
            "## Cost-scheme sensitivity",
            "",
            "Each sensitivity scheme rebuilds adaptive lambda candidates from train only, reselects one operating point on validation, and evaluates one locked point on test. The ordered threshold grid remains train-derived and is reselected by named-scheme validation cost.",
            "",
            "| Scheme | Policy | Selected parameter | Validation C-index | Test C-index | Mean test cost | Mean tests |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scheme, scheme_result in results["cost_scheme_sensitivity"].items():
        for policy in ("ordered", "adaptive"):
            record = scheme_result[policy]
            parameter = record["selected"]["parameter"]
            parameter_text = "infinity" if parameter is None else f"{float(parameter):.10g}"
            lines.append(
                f"| `{scheme}` | {policy} | {parameter_text} | "
                f"{_format_metric(record['selected']['validation_c_index'])} | "
                f"{_format_metric(record['test']['prediction']['harrell_c_index'])} | "
                f"{_format_metric(record['test']['acquisition']['mean_cost'], 3)} | "
                f"{_format_metric(record['test']['acquisition']['mean_acquired_tests'], 3)} |"
            )

    adaptive_vs_ordered = bootstrap["contrasts"][
        "adaptive_acquisition_minus_ordered_acquisition"
    ]
    lines.extend(
        [
            "",
            "## Ablations",
            "",
            "| Ablation/comparison | Consequence on the locked test split |",
            "|---|---|",
            f"| Without cost (`lambda=0`) | Acquires all six; identical final Cox risk and cost to Full Features (C-index {_format_metric(primary['full_features']['test']['prediction']['harrell_c_index'])}, cost 6.000). |",
            "| Without STOP | Must acquire all six; identical final Cox risk and cost to Full Features. |",
            f"| Fixed vs adaptive order | Adaptive minus ordered paired C-index: {_format_interval(adaptive_vs_ordered['harrell_c_index'])}; paired mean-cost difference: {_format_interval(adaptive_vs_ordered['mean_cost'], 3)}. |",
            f"| Ordered threshold sensitivity | {len(results['quality_cost_curves']['ordered_acquisition'])} train-derived thresholds evaluated on validation; one selected threshold evaluated on test. |",
            f"| Adaptive penalty sensitivity | {len(results['quality_cost_curves']['adaptive_acquisition'])} train-derived penalties evaluated on validation; one selected penalty evaluated on test. |",
            "",
            "## Interpretation",
            "",
            f"The selected adaptive policy uses {_format_metric(primary['adaptive_acquisition']['test']['acquisition']['mean_cost'], 3)} relative units on average versus {_format_metric(primary['ordered_acquisition']['test']['acquisition']['mean_cost'], 3)} for ordered NEXT/STOP and 6.000 for Full Features. Its test C-index is {_format_metric(primary['adaptive_acquisition']['test']['prediction']['harrell_c_index'])}. Static K=3 costs 3.000 and has test C-index {_format_metric(primary['static_top_k']['test']['prediction']['harrell_c_index'])}. Thus the evidence supports lower average acquisition burden, but it does not show that the current adaptive value surrogate dominates every static subset in predictive quality.",
            "",
            "## Limitations",
            "",
            "- The same single validation split informed static baseline orders and ordered acquisition-8 operating points; uncertainty from model and policy development is not captured by the test-only bootstrap.",
            "- The event rate is low, so PR-AUC and calibration estimates are more fragile than the sample size alone suggests.",
            "- Harrell C-index can be censoring-sensitive; the fixed-horizon AUC, PR-AUC, Brier, and calibration analyses complement it but do not replace external validation.",
            "- Relative costs are experimental weights, omit blood-draw/panel effects, and must not be interpreted as monetary or clinical burden estimates.",
            "- Requested source-missing laboratory results are charged and represented by train-fitted imputation, which is a retrospective simulation convention.",
            "- Absolute risk revision is not causal value, patient benefit, or a guarantee of lower prediction error.",
            "- Results come from one primary cohort and one internal test split. external validation external validation is still required.",
            "",
            "cost-performance evaluation makes no clinical deployment claim and writes no participant-level output.",
            "",
        ]
    )
    return "\n".join(lines)


def run_cost_performance_evaluation(
    *,
    project_root: Path,
    cohort_path: Path,
    cohort_manifest_path: Path,
    feature_config_path: Path,
    cost_config_path: Path,
    static_metrics_path: Path,
    static_model_path: Path,
    ordered_metrics_path: Path,
    adaptive_metrics_path: Path,
    adaptive_model_path: Path,
    metrics_path: Path,
    report_path: Path,
    cost_figure_path: Path,
    calibration_figure_path: Path,
    retention_target: float = PRIMARY_RETENTION_TARGET,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = EVALUATION_BOOTSTRAP_SEED,
    lambda_quantiles: int = DEFAULT_LAMBDA_QUANTILES,
) -> dict[str, Any]:

    if not 0.0 < retention_target <= 1.0:
        raise ValueError("retention_target must lie in (0, 1].")
    static_results = _read_json(static_metrics_path, 5)
    ordered_results = _read_json(ordered_metrics_path, 6)
    adaptive_results = _read_json(adaptive_metrics_path, 7)
    cohort_hash = sha256_file(cohort_path)
    for stage, document in ((5, static_results), (6, ordered_results), (7, adaptive_results)):
        if document["dataset"]["cohort_sha256"] != cohort_hash:
            raise ValueError(f"Stage {stage} metrics reference a different cohort.")
    if adaptive_results["adaptive_model_artifact"]["sha256"] != sha256_file(adaptive_model_path):
        raise ValueError("adaptive acquisition value-model artifact hash differs from its report.")

    design = load_feature_cost_design(feature_config_path, cost_config_path)
    costs = design.costs
    splits = load_cohort_splits(cohort_path)
    bundle = load_adaptive_model_bundle(static_metrics_path, static_model_path)
    prepared = {
        "train": prepare_adaptive_data(bundle, splits.train),
        "validation": prepare_adaptive_data(bundle, splits.validation),
        "test": prepare_adaptive_data(bundle, splits.test),
    }
    value_models = _load_value_models(adaptive_model_path)
    target = _validation_target(static_results, retention_target)
    target_c = target["target_validation_c_index"]

    top_k_point = _select_static_point(static_results, "static_top_k", target_c)
    greedy_point = _select_static_point(static_results, "importance_greedy", target_c)
    random_point = _select_random_point(static_results, target_c)
    top_k_mask = subset_to_mask(tuple(top_k_point["acquired_features"]))
    greedy_mask = subset_to_mask(tuple(greedy_point["acquired_features"]))
    if top_k_mask != greedy_mask:
        raise ValueError("cost-performance evaluation report assumes observed Top-K and Greedy selections coincide.")
    if int(random_point["acquired_count"]) != len(CORE_ACQUIRABLE_FEATURES):
        raise ValueError("Selected Random point is not the observed Full Features endpoint.")

    order = tuple(ordered_results["fixed_order"]["order"])
    prefix_masks = [subset_to_mask(order[:stage]) for stage in range(len(order) + 1)]
    stage_risks = {
        split: data.subset_risks[:, prefix_masks]
        for split, data in prepared.items()
    }
    threshold = _parameter(
        ordered_results["ordered_policy"]["selected_operating_point"]["threshold"]
    )
    ordered_simulations = {
        split: simulate_ordered_policy(risks, threshold)
        for split, risks in stage_risks.items()
        if split != "train"
    }
    penalty = _parameter(
        adaptive_results["adaptive_policy"]["selected_operating_point"]["cost_penalty"]
    )
    adaptive_simulations = {
        split: simulate_adaptive_policy(
            prepared[split], value_models, costs, penalty
        )
        for split in ("validation", "test")
    }

    full_mask = 2 ** len(CORE_ACQUIRABLE_FEATURES) - 1
    risks_by_split: dict[str, dict[str, NDArray[np.float64]]] = {}
    for split in ("validation", "test"):
        data = prepared[split]
        risks_by_split[split] = {
            "base_only": data.subset_risks[:, 0],
            "full_features": data.subset_risks[:, full_mask],
            "static_top_k": data.subset_risks[:, top_k_mask],
            "random_acquisition": data.subset_risks[:, full_mask],
            "importance_greedy": data.subset_risks[:, greedy_mask],
            "ordered_acquisition": ordered_simulations[split].selected_risk,
            "adaptive_acquisition": adaptive_simulations[split].selected_risk,
        }

    primary: dict[str, dict[str, Any]] = {}
    method_selection = {
        "base_only": "fixed endpoint; no selection",
        "full_features": "fixed endpoint; no selection",
        "static_top_k": f"minimum K={top_k_point['acquired_count']} meeting validation target",
        "random_acquisition": f"minimum K={random_point['acquired_count']} mean curve meeting validation target",
        "importance_greedy": f"minimum K={greedy_point['acquired_count']} meeting validation target",
        "ordered_acquisition": "minimum validation cost meeting target",
        "adaptive_acquisition": "minimum validation cost meeting target",
    }
    fixed_counts = {
        "base_only": 0,
        "full_features": 6,
        "static_top_k": int(top_k_point["acquired_count"]),
        "random_acquisition": int(random_point["acquired_count"]),
        "importance_greedy": int(greedy_point["acquired_count"]),
    }
    primary_cost_arrays: dict[str, NDArray[np.float64]] = {}
    for method in METHOD_LABELS:
        primary[method] = {"selection_summary": method_selection[method]}
        for split in ("validation", "test"):
            frame = getattr(splits, split)
            if method in fixed_counts:
                acquisition, participant_cost, acquired_count = _fixed_acquisition_metrics(
                    len(frame), fixed_counts[method], float(fixed_counts[method])
                )
            elif method == "ordered_acquisition":
                simulation = ordered_simulations[split]
                participant_cost = prefix_costs(costs, order)[simulation.stop_stage]
                acquired_count = simulation.stop_stage.astype(np.int64)
                acquisition = _acquisition_metrics(acquired_count, participant_cost)
            else:
                simulation = adaptive_simulations[split]
                participant_cost = _mask_costs(costs, costs.default_scheme)[
                    simulation.final_masks
                ]
                acquired_count = simulation.acquired_count
                acquisition = _acquisition_metrics(acquired_count, participant_cost)
            primary[method][split] = {
                "prediction": _prediction_metrics(
                    splits.train, frame, risks_by_split[split][method]
                ),
                "acquisition": acquisition,
            }
            if split == "test":
                primary_cost_arrays[method] = participant_cost

    expected = {
        "ordered_acquisition": float(
            ordered_results["evaluation"]["selected_policy"]["test"]["prediction"][
                "harrell_c_index"
            ]
        ),
        "adaptive_acquisition": float(
            adaptive_results["evaluation"]["selected_policy"]["test"]["prediction"][
                "harrell_c_index"
            ]
        ),
    }
    for method, reported in expected.items():
        reproduced = float(primary[method]["test"]["prediction"]["harrell_c_index"])
        if not math.isclose(reproduced, reported, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Reproduced {method} test result differs from its stage report.")

    bootstrap_methods = (
        "base_only",
        "full_features",
        "static_top_k",
        "ordered_acquisition",
        "adaptive_acquisition",
    )
    bootstrap = paired_bootstrap_evaluation(
        splits.train["duration_years"].to_numpy(dtype=float),
        splits.train["event"].to_numpy(dtype=np.int64),
        splits.test["duration_years"].to_numpy(dtype=float),
        splits.test["event"].to_numpy(dtype=np.int64),
        {method: risks_by_split["test"][method] for method in bootstrap_methods},
        {method: primary_cost_arrays[method] for method in bootstrap_methods},
        EVALUATION_HORIZON_YEARS,
        contrasts=(("adaptive_acquisition", "ordered_acquisition"),),
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    primary_curves = _primary_curves(static_results, ordered_results, adaptive_results)
    scheme_sensitivity: dict[str, Any] = {}
    for scheme in costs.schemes:
        scheme_sensitivity[scheme] = {
            "ordered": _ordered_scheme_evaluation(
                scheme=scheme,
                ordered_results=ordered_results,
                validation_stage_risks=stage_risks["validation"],
                test_stage_risks=stage_risks["test"],
                test_frame=splits.test,
                train_frame=splits.train,
                costs=costs,
                order=order,
                target=target_c,
            ),
            "adaptive": _adaptive_scheme_evaluation(
                scheme=scheme,
                train_data=prepared["train"],
                validation_data=prepared["validation"],
                test_data=prepared["test"],
                validation_frame=splits.validation,
                test_frame=splits.test,
                train_frame=splits.train,
                value_models=value_models,
                costs=costs,
                target=target_c,
                lambda_quantiles=lambda_quantiles,
            ),
        }

    full_test = primary["full_features"]["test"]
    ablations = {
        "without_cost": {
            "definition": "adaptive utility with lambda = 0",
            "result": "all six candidates acquired; final risk equals Full Features",
            "test": full_test,
        },
        "without_stop": {
            "definition": "remove STOP and continue until no candidate remains",
            "result": "all six candidates acquired; final risk equals Full Features",
            "test": full_test,
        },
        "fixed_vs_adaptive_order": {
            "ordered": primary["ordered_acquisition"]["test"],
            "adaptive": primary["adaptive_acquisition"]["test"],
            "paired_bootstrap_contrast": bootstrap["contrasts"][
                "adaptive_acquisition_minus_ordered_acquisition"
            ],
        },
        "threshold_and_penalty_sensitivity": {
            "ordered_validation_points": len(primary_curves["ordered_acquisition"]),
            "adaptive_validation_points": len(primary_curves["adaptive_acquisition"]),
            "test_points_per_primary_policy": 1,
        },
    }

    results: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 8,
        "scope": "aggregate cost-performance evaluation of locked baseline modelling-7 methods",
        "seeds": {
            "cohort_split": DEFAULT_SEED,
            "bootstrap": bootstrap_seed,
            "adaptive_train_state_sampling": value_models.seed,
        },
        "dataset": {
            "cohort_path": portable_path(cohort_path, project_root),
            "cohort_sha256": cohort_hash,
            "cohort_manifest_path": portable_path(cohort_manifest_path, project_root),
            "cohort_manifest_sha256": sha256_file(cohort_manifest_path),
            "test_participants": int(len(splits.test)),
            "test_events": int(splits.test["event"].sum()),
        },
        "provenance": {
            "static_metrics": {
                "path": portable_path(static_metrics_path, project_root),
                "sha256": sha256_file(static_metrics_path),
            },
            "static_local_model": {
                "path": portable_path(static_model_path, project_root),
                "sha256": sha256_file(static_model_path),
                "committed": False,
            },
            "ordered_metrics": {
                "path": portable_path(ordered_metrics_path, project_root),
                "sha256": sha256_file(ordered_metrics_path),
            },
            "adaptive_metrics": {
                "path": portable_path(adaptive_metrics_path, project_root),
                "sha256": sha256_file(adaptive_metrics_path),
            },
            "adaptive_local_model": {
                "path": portable_path(adaptive_model_path, project_root),
                "sha256": sha256_file(adaptive_model_path),
                "committed": False,
            },
        },
        "features": {
            "base": list(BASE_FEATURES),
            "acquisition_candidates": list(CORE_ACQUIRABLE_FEATURES),
            "optional_not_used": list(OPTIONAL_FEATURES),
            "excluded_from_predictors_and_policy_state": list(
                EXCLUDED_FROM_MAIN_PREDICTORS
            ),
        },
        "selection_contract": {
            **target,
            "selection_split": "validation",
            "test_used_for_selection": False,
            "test_use": "one post-lock evaluation per pre-specified primary or cost-sensitivity policy",
        },
        "evaluation": {
            "horizon_years": EVALUATION_HORIZON_YEARS,
            "harrell_c_index": "all observed follow-up; higher risk means earlier event",
            "ipcw_brier": "train-derived censoring distribution; censored by horizon contributes zero",
            "ipcw_auc": "cumulative cases by horizon and dynamic controls beyond horizon; train-derived IPCW",
            "ipcw_pr_auc": "IPCW average precision on the same cumulative/dynamic horizon set",
            "calibration": f"{CALIBRATION_BIN_COUNT} equal-count rank bins with Kaplan-Meier observed risk",
        },
        "primary_cost_scheme": costs.default_scheme,
        "primary_comparison": primary,
        "quality_cost_curves": primary_curves,
        "bootstrap": bootstrap,
        "cost_scheme_sensitivity": scheme_sensitivity,
        "ablations": ablations,
        "figures": {
            "cost_performance": portable_path(cost_figure_path, project_root),
            "test_calibration": portable_path(calibration_figure_path, project_root),
        },
        "participant_level_output": None,
        "new_model_fitted_in_evaluation": False,
        "environment": package_versions(),
    }

    _write_figures(
        primary_curves,
        primary,
        target_c,
        cost_figure_path,
        calibration_figure_path,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_evaluation_report(results), encoding="utf-8")
    return results
