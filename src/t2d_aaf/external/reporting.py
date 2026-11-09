from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from t2d_aaf.evaluation.cost_performance import (
    paired_bootstrap_evaluation,
    prediction_evaluation,
)
from t2d_aaf.external.nagala import (
    COMPATIBLE_ACQUIRABLE_FEATURES,
    COMPATIBLE_BASE_FEATURES,
    DATASET_DOI,
    UNAVAILABLE_ACQUIRABLE_FEATURES,
    UNAVAILABLE_BASE_FEATURES,
    NagalaCohort,
    load_nagala_cohort,
)
from t2d_aaf.external.validation import (
    ExternalMethodVectors,
    acquisition_summary,
    build_external_prediction_run,
)
from t2d_aaf.features.registry import load_feature_cost_design
from t2d_aaf.survival.baseline import EVALUATION_HORIZON_YEARS
from t2d_aaf.survival.reporting import package_versions, portable_path, sha256_file


EXTERNAL_SCHEMA_VERSION = "nagala_external_validation_v1"
EXTERNAL_BOOTSTRAP_SEED = 20260827
DEFAULT_EXTERNAL_BOOTSTRAP_REPLICATES = 200
CALIBRATION_BIN_COUNT = 10

METHOD_LABELS: Mapping[str, str] = {
    "base_only": "Base Only (family history imputed)",
    "compatible_full": "All Compatible Labs",
    "static_top_k": "Static Top-K",
    "importance_greedy": "Importance Greedy",
    "ordered_compatible": "Ordered Compatible",
    "adaptive_compatible": "Adaptive Compatible",
}


def _read_report(path: Path, stage: int) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("stage") != stage:
        raise ValueError(f"Expected Stage {stage} metrics at {path}.")
    return document


def _external_prediction_metrics(
    cohort: NagalaCohort,
    vectors: ExternalMethodVectors,
) -> dict[str, object]:
    duration = cohort.mapped["duration_years"].to_numpy(dtype=float)
    event = cohort.mapped["event"].to_numpy(dtype=np.int64)
    return prediction_evaluation(
        duration,
        event,
        duration,
        event,
        vectors.predicted_risk,
        EVALUATION_HORIZON_YEARS,
        calibration_bin_count=CALIBRATION_BIN_COUNT,
    )


def _write_figures(
    method_results: Mapping[str, Mapping[str, Any]],
    cost_figure_path: Path,
    calibration_figure_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cost_figure_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "base_only": "#777777",
        "compatible_full": "#222222",
        "static_top_k": "#25706f",
        "importance_greedy": "#d07a30",
        "ordered_compatible": "#4063a5",
        "adaptive_compatible": "#a33f55",
    }
    markers = {
        "base_only": "o",
        "compatible_full": "s",
        "static_top_k": "^",
        "importance_greedy": "v",
        "ordered_compatible": "D",
        "adaptive_compatible": "P",
    }
    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for method, record in method_results.items():
        axis.scatter(
            float(record["acquisition"]["mean_cost"]),
            float(record["prediction"]["harrell_c_index"]),
            color=colors[method],
            marker=markers[method],
            s=70,
            label=METHOD_LABELS[method],
            zorder=3,
        )
    axis.set_xlabel("Mean compatible-feature cost on NAGALA")
    axis.set_ylabel("External Harrell C-index")
    axis.set_xlim(left=-0.15)
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(cost_figure_path, dpi=180)
    plt.close(figure)

    calibration_methods = (
        "base_only",
        "compatible_full",
        "static_top_k",
        "ordered_compatible",
        "adaptive_compatible",
    )
    figure, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    maximum = 0.0
    for method in calibration_methods:
        bins = method_results[method]["prediction"]["calibration_bins"]
        predicted = [float(row["mean_predicted_risk"]) for row in bins]
        observed = [float(row["km_observed_risk"]) for row in bins]
        maximum = max(maximum, *predicted, *observed)
        axis.plot(
            predicted,
            observed,
            color=colors[method],
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
    axis.set_ylabel("NAGALA Kaplan-Meier observed 3-year risk")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8)
    figure.savefig(calibration_figure_path, dpi=180)
    plt.close(figure)


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _interval(interval: Mapping[str, Any], digits: int = 4) -> str:
    return (
        f"{float(interval['point_estimate']):.{digits}f} "
        f"[{float(interval['ci95_lower']):.{digits}f}, "
        f"{float(interval['ci95_upper']):.{digits}f}]"
    )


def render_nagala_profile(audit: Mapping[str, Any]) -> str:
    outcome = audit["outcome"]
    integrity = audit["integrity"]
    workbook = audit["workbook"]
    lines = [
        "# NAGALA Dataset Profile",
        "",
        "## Source and integrity",
        "",
        f"- Dataset DOI: `https://doi.org/{audit['dataset_doi']}`.",
        f"- Workbook: `data/external/{audit['file_name']}`.",
        f"- File size: {audit['file_size_bytes']:,} bytes.",
        f"- Dryad MD5: `{audit['md5']}`.",
        f"- SHA-256: `{audit['sha256']}`.",
        "- The local file was obtained from the Zenodo mirror record `4996032` because the Dryad public file endpoint returned HTTP 403; the byte size and MD5 exactly match Dryad API metadata.",
        "- Raw external data is immutable, ignored by Git, and never written to reports at participant level.",
        "",
        "## Real workbook structure",
        "",
        f"- Sheets: `{workbook['sheets'][0]}` only.",
        f"- Data rows: {workbook['data_rows']:,}, plus one header row.",
        f"- Columns: {workbook['columns']}.",
        f"- Row meaning: {workbook['row_meaning']}.",
        f"- Duplicate full rows: {integrity['duplicate_rows']:,}.",
        f"- Missing/duplicate patient numbers: {integrity['missing_identifiers']:,}/{integrity['duplicate_identifiers']:,}.",
        f"- Rows excluded: {integrity['rows_excluded']:,}.",
        "",
        "## Columns",
        "",
        "| # | Exact raw column | Group | Dtype | Missing | Missing % | Unique | Min | Median | Max | Mapped to |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for profile in audit["column_profiles"]:
        summary = profile.get("summary", {})
        mapped_to = f"`{profile['mapped_to']}`" if profile["mapped_to"] else "-"
        lines.append(
            f"| {profile['position']} | `{_md(profile['name'])}` | {profile['group']} | "
            f"`{profile['dtype']}` | {profile['missing']:,} | {_number(profile['missing_percent'], 3)} | "
            f"{profile['unique_non_missing']:,} | {_number(summary['minimum']) if summary else '-'} | "
            f"{_number(summary['median']) if summary else '-'} | {_number(summary['maximum']) if summary else '-'} | "
            f"{mapped_to} |"
        )

    lines.extend(
        [
            "",
            "## Outcome, follow-up, and censoring",
            "",
            f"- Event: `{outcome['event_raw_column']}`, with `{outcome['event_mapping']}`.",
            f"- Events: {outcome['events']:,} ({_number(outcome['event_percent'], 2)}%).",
            f"- Duration: `{outcome['duration_raw_column']}`; `{outcome['duration_mapping']}`.",
            f"- Duration range: {_number(outcome['duration_years']['minimum'], 3)} to {_number(outcome['duration_years']['maximum'], 3)} years; median {_number(outcome['duration_years']['median'], 3)} years.",
            f"- Censoring: {outcome['censoring']}.",
            "- There are no missing or non-positive durations and no invalid event codes.",
            "",
            "## Confirmed schema issues",
            "",
            "- Sex header mismatch: the header claims codes 1/2, but the file contains 0/1. Published counts confirm raw `1 = male` and raw `0 = female`; mapping to the primary coding is `1 -> 1`, `0 -> 2`.",
            "- `Obesity phenotype 1-8` contains 15 observed codes, including 0/1/10/11/100/101/110/111. It is not mapped or used.",
            "- HDL has 11 missing mg/dL values represented as zero in the parallel mmol/L column. HDL is optional and unused in external validation.",
            "- Provided mmol/L and mg/dL versions of FPG, total cholesterol, and triglycerides agree within rounding tolerance.",
            "",
            "## Primary feature compatibility",
            "",
            "| Primary feature | NAGALA source | Status | external validation handling |",
            "|---|---|---|---|",
            "| `age` | `Baseline Age, yrs` | compatible | direct |",
            "| `gender` | `Sex, 1:Men, 2:Women` | compatible after coding correction | raw 1 to primary 1; raw 0 to primary 2 |",
            "| `bmi` | `Baseline BMI, kg/m2` | compatible | direct |",
            "| `sbp` | `Baseline Systolic blood pressure, mmHg` | compatible | direct |",
            "| `dbp` | `Baseline Diastolic blood pressure, mmHg` | compatible | direct |",
            "| `family_history_diabetes` | absent | unavailable base | source-missing; primary train-fitted imputation |",
            "| `fpg` | `Baseline Fasting plasma glucose, mmol/L` | compatible action | direct |",
            "| `total_cholesterol` | `Baseline Total Cholesterol, mmol/L` | compatible action | direct |",
            "| `triglycerides` | `Baseline Triglycerides, mmol/L` | compatible action | direct |",
            "| `alt` | `Baseline ALT, IU/L` | compatible action | treated as numerically equivalent to primary U/L |",
            "| `ccr` | absent | unavailable action | removed from external action set |",
            "| `bun` | absent | unavailable action | removed from external action set |",
            "",
            "The exact primary Full Features model and the selected K=6 Random Acquisition endpoint cannot be externally evaluated because CCR and BUN are absent. external validation is therefore a partial compatible-feature validation, not a complete replication of cost-performance evaluation.",
            "",
            "## Continuous-feature audit",
            "",
            "| Feature | Min | P01 | P25 | Median | P75 | P99 | Max | Audit range | Below | Above | Action |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for row in audit["clinical_audit"]:
        lines.append(
            f"| `{row['feature']}` | {_number(row['minimum'], 3)} | {_number(row['p01'], 3)} | "
            f"{_number(row['p25'], 3)} | {_number(row['median'], 3)} | {_number(row['p75'], 3)} | "
            f"{_number(row['p99'], 3)} | {_number(row['maximum'], 3)} | "
            f"{_number(row['audit_lower'], 2)}-{_number(row['audit_upper'], 2)}; {row['audit_rationale']} | "
            f"{row['below']:,} | {row['above']:,} | {row['action']} |"
        )
    lines.extend(
        [
            "",
            "No mapped row or continuous value is deleted, clipped, winsorized, or corrected. The primary train-fitted preprocessing handles only the explicitly absent family-history field.",
            "",
            "## Leakage and suitability",
            "",
            "- `Patient Number` is used only for uniqueness checks and never enters a predictor or policy state.",
            "- `Incident DM 0/1` and `Follow up duration, days` are outcomes used only for external evaluation.",
            "- Obesity phenotypes, fatty liver, HbA1c, lifestyle fields, and duplicate-unit columns are not added to the primary predictor contract.",
            "- One row per unique participant, positive follow-up, a valid incident event, and sufficient follow-up beyond three years support survival and fixed-horizon evaluation.",
            "- The dataset is suitable for partial external validation of Base Only, the compatible laboratory subset, locked Static Top-K, and compatible constrained ordered/adaptive policies.",
            "- It is not suitable for exact external validation of the six-action Full Features or Random K=6 endpoints.",
            "",
        ]
    )
    return "\n".join(lines)


def render_external_results(results: Mapping[str, Any]) -> str:
    methods = results["methods"]
    bootstrap = results["bootstrap"]
    bootstrap_alias = {
        "base_only": "base_only",
        "compatible_full": "compatible_full",
        "static_top_k": "static_top_k",
        "importance_greedy": "static_top_k",
        "ordered_compatible": "ordered_compatible",
        "adaptive_compatible": "adaptive_compatible",
    }
    lines = [
        "# external validation - NAGALA External Validation",
        "",
        "## Protocol lock",
        "",
        "This is a partial compatible-feature validation of primary-trained models and locked policies. NAGALA outcomes are never used to select a subset, threshold, lambda, action cost, or recalibration parameter.",
        "",
        f"- Participants: {results['external_dataset']['participants']:,}.",
        f"- Incident diabetes events: {results['external_dataset']['events']:,} ({_number(results['external_dataset']['event_percent'], 2)}%).",
        f"- Horizon: {results['evaluation']['horizon_years']:.1f} years.",
        f"- Bootstrap: {bootstrap['replicates_requested']} event-stratified paired replicates, seed `{bootstrap['seed']}`.",
        "- The external cohort's censoring distribution is used only to compute external IPCW metrics.",
        f"- The 3-year IPCW analysis contains {methods['base_only']['prediction']['horizon_cases']:,} cases and {methods['base_only']['prediction']['horizon_controls']:,} dynamic controls; {methods['base_only']['prediction']['excluded_censored_before_or_at_horizon']:,} participants censored by the horizon are excluded from AUC/PR-AUC.",
        "- Family history is absent and receives the primary train-fitted imputation; CCR and BUN are not available actions.",
        "- No participant-level output is written.",
        "",
        "## External performance",
        "",
        "| Method | External C-index (95% CI) | IPCW Brier@3y (95% CI) | IPCW AUC@3y (95% CI) | IPCW PR-AUC@3y (95% CI) | Mean cost (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, label in METHOD_LABELS.items():
        intervals = bootstrap["methods"][bootstrap_alias[method]]
        lines.append(
            f"| {label} | {_interval(intervals['harrell_c_index'])} | "
            f"{_interval(intervals['ipcw_brier_at_horizon'])} | "
            f"{_interval(intervals['ipcw_cumulative_dynamic_auc'])} | "
            f"{_interval(intervals['ipcw_average_precision'])} | "
            f"{_interval(intervals['mean_cost'], 3)} |"
        )

    lines.extend(
        [
            "",
            "![External performance versus compatible cost](figures/external_cost_performance.png)",
            "",
            "## Acquisition burden",
            "",
            "| Method | Mean tests | Median tests | STOP at base | Acquire all four compatible labs | Mean cost |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, label in METHOD_LABELS.items():
        acquisition = methods[method]["acquisition"]
        lines.append(
            f"| {label} | {_number(acquisition['mean_acquired_tests'], 3)} | "
            f"{_number(acquisition['median_acquired_tests'], 1)} | "
            f"{_number(acquisition['stop_without_labs_percent'], 2)}% | "
            f"{_number(acquisition['acquire_all_compatible_percent'], 2)}% | "
            f"{_number(acquisition['mean_cost'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Calibration",
            "",
            "| Method | Mean predicted risk | KM observed risk | Calibration-in-large | Mean absolute bin gap | Maximum bin gap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, label in METHOD_LABELS.items():
        prediction = methods[method]["prediction"]
        lines.append(
            f"| {label} | {_number(prediction['mean_predicted_risk_at_horizon'])} | "
            f"{_number(prediction['km_observed_risk_at_horizon'])} | "
            f"{_number(prediction['calibration_in_the_large_difference'])} | "
            f"{_number(prediction['calibration_mean_absolute_bin_gap'])} | "
            f"{_number(prediction['calibration_maximum_absolute_bin_gap'])} |"
        )
    lines.extend(
        [
            "",
            "![NAGALA calibration](figures/external_calibration.png)",
            "",
            "## What cannot be evaluated exactly",
            "",
            "- Primary Full Features requires CCR and BUN, which NAGALA does not contain. `All Compatible Labs` is a four-laboratory surrogate, not Full Features.",
            "- The cost-performance evaluation selected Random Acquisition endpoint is K=6 and therefore cannot be mapped.",
            "- Ordered and adaptive policies are constrained to four available actions without retuning. They test transport under a reduced action space, not exact seven-action policy replication.",
            "- Family history is unavailable for every NAGALA participant, so Base Only transport includes deterministic primary-train imputation for that field.",
            "",
            "## Interpretation",
            "",
            f"The compatible adaptive policy uses {_number(methods['adaptive_compatible']['acquisition']['mean_cost'], 3)} tests/cost units on average, compared with {_number(methods['ordered_compatible']['acquisition']['mean_cost'], 3)} for the compatible ordered policy and 4.000 for all compatible laboratories. Its external C-index is {_number(methods['adaptive_compatible']['prediction']['harrell_c_index'])}. Adaptive minus ordered C-index is {_interval(bootstrap['contrasts']['adaptive_compatible_minus_ordered_compatible']['harrell_c_index'])}, while the paired mean-cost difference is {_interval(bootstrap['contrasts']['adaptive_compatible_minus_ordered_compatible']['mean_cost'], 3)}. The locked Static Top-K C-index decreases from {_number(results['primary_internal_references']['static_top_k']['harrell_c_index'])} on the primary test split to {_number(methods['static_top_k']['prediction']['harrell_c_index'])} on NAGALA. The adaptive policy therefore transports with substantially lower compatible cost but a small measurable discrimination loss versus ordered acquisition; none of these results establishes clinical utility or complete validation of the original six-action policy.",
            "",
            "## Limitations",
            "",
            "- NAGALA and the primary Chinese cohort differ in country, recruitment, follow-up, event rate, and covariate distribution.",
            "- Three primary predictors are absent, including two acquisition actions.",
            "- ALT is treated as numerically compatible between IU/L and U/L.",
            "- Bootstrap intervals capture external participant sampling only; primary model/policy development uncertainty is not propagated.",
            "- No recalibration is performed. Any calibration shift is reported rather than corrected using external outcomes.",
            "- Relative costs remain experimental units and do not represent Japanese clinical prices or burden.",
            "",
            "external validation writes no external participant rows, predictions, or paths and makes no deployment claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run_external_validation(
    *,
    project_root: Path,
    external_path: Path,
    feature_config_path: Path,
    cost_config_path: Path,
    static_metrics_path: Path,
    static_model_path: Path,
    ordered_metrics_path: Path,
    adaptive_metrics_path: Path,
    adaptive_model_path: Path,
    evaluation_metrics_path: Path,
    profile_path: Path,
    metrics_path: Path,
    report_path: Path,
    cost_figure_path: Path,
    calibration_figure_path: Path,
    bootstrap_replicates: int = DEFAULT_EXTERNAL_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = EXTERNAL_BOOTSTRAP_SEED,
) -> dict[str, Any]:

    static_results = _read_report(static_metrics_path, 5)
    ordered_results = _read_report(ordered_metrics_path, 6)
    adaptive_results = _read_report(adaptive_metrics_path, 7)
    evaluation_results = _read_report(evaluation_metrics_path, 8)
    if evaluation_results["provenance"]["static_metrics"]["sha256"] != sha256_file(
        static_metrics_path
    ):
        raise ValueError("cost-performance evaluation references different static baseline metrics.")
    if evaluation_results["provenance"]["ordered_metrics"]["sha256"] != sha256_file(
        ordered_metrics_path
    ):
        raise ValueError("cost-performance evaluation references different ordered acquisition metrics.")
    if evaluation_results["provenance"]["adaptive_metrics"]["sha256"] != sha256_file(
        adaptive_metrics_path
    ):
        raise ValueError("cost-performance evaluation references different adaptive acquisition metrics.")
    if evaluation_results["provenance"]["static_local_model"]["sha256"] != sha256_file(
        static_model_path
    ):
        raise ValueError("static baseline local model differs from cost-performance evaluation provenance.")
    if evaluation_results["provenance"]["adaptive_local_model"]["sha256"] != sha256_file(
        adaptive_model_path
    ):
        raise ValueError("adaptive acquisition local model differs from cost-performance evaluation provenance.")

    design = load_feature_cost_design(feature_config_path, cost_config_path)
    cohort = load_nagala_cohort(external_path)
    prediction_run = build_external_prediction_run(
        frame=cohort.mapped,
        static_metrics_path=static_metrics_path,
        static_model_path=static_model_path,
        static_results=static_results,
        ordered_results=ordered_results,
        adaptive_results=adaptive_results,
        evaluation_results=evaluation_results,
        adaptive_model_path=adaptive_model_path,
        costs=design.costs,
    )

    method_results: dict[str, dict[str, Any]] = {}
    for method, vectors in prediction_run.methods.items():
        method_results[method] = {
            **dict(prediction_run.method_metadata[method]),
            "prediction": _external_prediction_metrics(cohort, vectors),
            "acquisition": acquisition_summary(vectors),
        }

    duration = cohort.mapped["duration_years"].to_numpy(dtype=float)
    event = cohort.mapped["event"].to_numpy(dtype=np.int64)
    bootstrap_methods = (
        "base_only",
        "compatible_full",
        "static_top_k",
        "ordered_compatible",
        "adaptive_compatible",
    )
    bootstrap = paired_bootstrap_evaluation(
        duration,
        event,
        duration,
        event,
        {
            method: prediction_run.methods[method].predicted_risk
            for method in bootstrap_methods
        },
        {
            method: prediction_run.methods[method].participant_cost
            for method in bootstrap_methods
        },
        EVALUATION_HORIZON_YEARS,
        contrasts=(
            ("adaptive_compatible", "ordered_compatible"),
            ("adaptive_compatible", "static_top_k"),
        ),
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    audit = cohort.audit
    results: dict[str, Any] = {
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 9,
        "scope": "partial compatible-feature external validation on NAGALA",
        "external_dataset": {
            "doi": DATASET_DOI,
            "path": portable_path(external_path, project_root),
            "file_size_bytes": audit["file_size_bytes"],
            "md5": audit["md5"],
            "sha256": audit["sha256"],
            "participants": int(len(cohort.mapped)),
            "events": int(event.sum()),
            "event_percent": float(100.0 * event.mean()),
            "profile_path": portable_path(profile_path, project_root),
            "raw_file_committed": False,
        },
        "primary_provenance": {
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
            "evaluation_metrics": {
                "path": portable_path(evaluation_metrics_path, project_root),
                "sha256": sha256_file(evaluation_metrics_path),
            },
        },
        "mapping": {
            "compatible_base_features": list(COMPATIBLE_BASE_FEATURES),
            "unavailable_base_features": list(UNAVAILABLE_BASE_FEATURES),
            "compatible_acquirable_features": list(COMPATIBLE_ACQUIRABLE_FEATURES),
            "unavailable_acquirable_features": list(
                UNAVAILABLE_ACQUIRABLE_FEATURES
            ),
            "gender": "NAGALA raw 1 -> primary 1 male; raw 0 -> primary 2 female",
            "duration": "NAGALA days / 365.25 -> years",
            "family_history": "all source-missing; primary train-fitted imputation",
            "unavailable_actions": "removed from the external action space; never acquired",
        },
        "evaluation": {
            "horizon_years": EVALUATION_HORIZON_YEARS,
            "external_outcomes_used_for_selection_or_recalibration": False,
            "external_outcomes_used_for_metrics_only": True,
            "censoring_distribution": "estimated on NAGALA for external IPCW metrics only",
            "calibration": f"{CALIBRATION_BIN_COUNT} equal-count bins; no recalibration",
            "primary_cost_scheme": design.costs.default_scheme,
            "policy_metadata": dict(prediction_run.policy_metadata),
        },
        "methods": method_results,
        "primary_internal_references": {
            method: {
                "harrell_c_index": float(
                    evaluation_results["primary_comparison"][method]["test"]["prediction"][
                        "harrell_c_index"
                    ]
                ),
                "ipcw_brier_at_horizon": float(
                    evaluation_results["primary_comparison"][method]["test"]["prediction"][
                        "ipcw_brier_at_horizon"
                    ]
                ),
                "mean_cost": float(
                    evaluation_results["primary_comparison"][method]["test"]["acquisition"][
                        "mean_cost"
                    ]
                ),
            }
            for method in ("base_only", "static_top_k")
        },
        "bootstrap": bootstrap,
        "not_evaluated_exactly": {
            "full_features": "CCR and BUN are absent",
            "random_acquisition": "the locked cost-performance evaluation selected endpoint is K=6",
            "unconstrained_ordered": "the final two fixed-order actions are absent",
            "unconstrained_adaptive": "two of six candidate actions are absent",
        },
        "external_data_audit": audit,
        "figures": {
            "cost_performance": portable_path(cost_figure_path, project_root),
            "calibration": portable_path(calibration_figure_path, project_root),
        },
        "participant_level_output": None,
        "external_rows_excluded": 0,
        "model_or_policy_refit_on_external": False,
        "environment": package_versions(),
    }

    _write_figures(method_results, cost_figure_path, calibration_figure_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(render_nagala_profile(audit), encoding="utf-8")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_external_results(results), encoding="utf-8")
    return results
