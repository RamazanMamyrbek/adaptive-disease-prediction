from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any

from t2d_aaf.data.cohort import (
    BASE_FEATURES,
    CORE_ACQUIRABLE_FEATURES,
    DEFAULT_SEED,
    EXCLUDED_FROM_MAIN_PREDICTORS,
)
from t2d_aaf.survival.baseline import (
    CONTINUOUS_FEATURES,
    EVALUATION_HORIZON_YEARS,
    MODEL_NAME,
    MODEL_TYPE,
    PREDICTOR_FEATURES,
    SPLIT_ORDER,
    TIES_METHOD,
    FittedCoxBaseline,
    coefficient_summary,
    evaluate_splits,
    fit_cox_baseline,
    load_cohort_splits,
    preprocessing_summary,
    save_model_artifact,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    distributions = ("numpy", "pandas", "scikit-learn", "statsmodels", "joblib")
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in distributions},
    }


def portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_results(
    *,
    project_root: Path,
    cohort_path: Path,
    cohort_manifest_path: Path,
    fitted: FittedCoxBaseline,
    metrics: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    return {
        "schema_version": "ordinary_survival_baseline_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 3,
        "model_name": MODEL_NAME,
        "scope": "ordinary survival baseline; no acquisition policy or cost model",
        "seed": DEFAULT_SEED,
        "dataset": {
            "cohort_path": portable_path(cohort_path, project_root),
            "cohort_sha256": sha256_file(cohort_path),
            "cohort_manifest_path": portable_path(cohort_manifest_path, project_root),
            "cohort_manifest_sha256": sha256_file(cohort_manifest_path),
        },
        "target": {"event": "event", "duration": "duration_years", "censoring": "event = 0"},
        "predictor_features": list(PREDICTOR_FEATURES),
        "base_features": list(BASE_FEATURES),
        "core_acquirable_features_used_as_ordinary_predictors": list(CORE_ACQUIRABLE_FEATURES),
        "excluded_from_predictors": list(EXCLUDED_FROM_MAIN_PREDICTORS),
        "model": {
            "type": MODEL_TYPE,
            "ties": TIES_METHOD,
            "penalization": None,
            "fit_split": "train",
            "fit_warnings": list(fitted.fit_warnings),
            "coefficients": coefficient_summary(fitted),
        },
        "preprocessing": preprocessing_summary(fitted.preprocessor),
        "evaluation": {
            "horizon_years": EVALUATION_HORIZON_YEARS,
            "c_index": "Harrell; higher linear predictor means higher risk",
            "brier": "single-horizon IPCW; censoring distribution estimated on train only",
            "observed_risk": "1 - Kaplan-Meier survival within each split",
            "metrics": metrics,
        },
        "cost_configuration": None,
        "acquisition_policy": None,
        "environment": package_versions(),
    }


def write_results_json(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def _format_metric(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _format_p_value(value: float) -> str:
    if value <= 0.0:
        return "<1e-300"
    return f"{value:.3g}"


def build_markdown_report(results: dict[str, Any]) -> str:
    metrics = results["evaluation"]["metrics"]
    preprocessing = results["preprocessing"]
    coefficients = results["model"]["coefficients"]
    lines = [
        "# baseline modelling Ordinary Survival Baseline",
        "",
        "This report records the ordinary future-T2D survival baseline. No Active Feature Acquisition policy, test cost, ordered acquisition, or adaptive acquisition is implemented in baseline modelling.",
        "",
        "## Experiment Contract",
        "",
        f"- Model: `{results['model_name']}` ({results['model']['type']})",
        f"- Cohort: `{results['dataset']['cohort_path']}`",
        f"- Cohort SHA-256: `{results['dataset']['cohort_sha256']}`",
        f"- Split seed inherited from cohort construction: `{results['seed']}`",
        "- Fit split: `train` only",
        "- Event: `event = 1`",
        "- Right-censoring: `event = 0` at `duration_years`",
        "- Evaluation horizon: `3.0 years`",
        "- Cox tie handling: `breslow`",
        "- Penalization: none",
        "",
        "## Predictor Set",
        "",
        "The baseline uses all cohort construction main predictors as an ordinary fixed feature vector:",
        "",
        "- Base: " + ", ".join(f"`{item}`" for item in results["base_features"]),
        "- Core laboratory predictors: "
        + ", ".join(
            f"`{item}`" for item in results["core_acquirable_features_used_as_ordinary_predictors"]
        ),
        "",
        "The laboratory variables are not acquired sequentially here. They are used together only to establish an ordinary full-information survival reference for later stages.",
        "",
        "Excluded from predictors: `participant_id`, `site`, `event`, `duration_years`, final-visit FPG, and the follow-up diagnosis flag. Optional cohort construction fields are not used.",
        "",
        "## Preprocessing And Leakage Control",
        "",
        "- Continuous variables: training median imputation, then training mean/standard-deviation scaling.",
        "- Binary coded variables: training mode imputation, no scaling.",
        "- No missingness indicators were added.",
        "- Validation and test values never contribute to imputation, scaling, fitting, baseline hazard, or censoring-distribution estimation.",
        "- No participant-level predictions are written to the repository.",
        "",
        "## Metrics",
        "",
        "| Split | N | Events | C-index | IPCW Brier @ 3y | Mean predicted risk @ 3y | KM observed risk @ 3y | Predicted - observed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLIT_ORDER:
        row = metrics[split]
        lines.append(
            f"| {split} | {row['participants']:,} | {row['events_total']:,} | "
            f"{_format_metric(row['harrell_c_index'])} | "
            f"{_format_metric(row['ipcw_brier_at_horizon'])} | "
            f"{_format_metric(row['mean_predicted_risk_at_horizon'])} | "
            f"{_format_metric(row['km_observed_risk_at_horizon'])} | "
            f"{_format_metric(row['calibration_in_the_large_difference'])} |"
        )

    lines.extend(
        [
            "",
            "Harrell's C-index evaluates ranking while respecting censoring. The 3-year Brier score uses inverse-probability-of-censoring weights estimated from the training split. The observed 3-year risk is estimated as `1 - Kaplan-Meier survival` within each split.",
            "",
            "### Horizon Support",
            "",
            "| Split | Events by 3y | Censored by 3y | Observed beyond 3y |",
            "|---|---:|---:|---:|",
        ]
    )
    for split in SPLIT_ORDER:
        row = metrics[split]
        lines.append(
            f"| {split} | {row['events_by_horizon']:,} | {row['censored_by_horizon']:,} | "
            f"{row['observed_beyond_horizon']:,} |"
        )

    lines.extend(
        [
            "",
            "## Cox Coefficients",
            "",
            "Continuous-feature hazard ratios are per one training standard deviation after imputation and scaling. Gender retains the source coding (`1 = male`, `2 = female`), so its hazard ratio is female versus male. Family history is coded `0/1`.",
            "",
            "| Feature | Coefficient | Hazard ratio | HR 95% CI | P-value |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in coefficients:
        lines.append(
            f"| {row['feature']} | {_format_metric(row['coefficient'])} | "
            f"{_format_metric(row['hazard_ratio'])} | "
            f"{_format_metric(row['hazard_ratio_ci95_low'])}-{_format_metric(row['hazard_ratio_ci95_high'])} | "
            f"{_format_p_value(row['p_value'])} |"
        )

    lines.extend(
        [
            "",
            "## Training Preprocessing Parameters",
            "",
            "| Continuous feature | Imputation median | Post-imputation mean | Scale |",
            "|---|---:|---:|---:|",
        ]
    )
    for feature in CONTINUOUS_FEATURES:
        row = preprocessing["continuous"][feature]
        lines.append(
            f"| {feature} | {_format_metric(row['imputation_median'])} | "
            f"{_format_metric(row['post_imputation_mean'])} | {_format_metric(row['post_imputation_scale'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This stage asks only whether a conventional censoring-aware model can extract prognostic signal from the confirmed baseline variables. The metrics are a technical baseline for later comparisons, not evidence that the model is ready for clinical use.",
            "",
            "## Limitations",
            "",
            "- The proportional-hazards assumption is not formally tested in this stage.",
            "- Confidence intervals for predictive metrics are not yet estimated.",
            "- The event rate is low (about 1.97%), and follow-up is relatively short and heterogeneous.",
            "- Median imputation is a simple baseline choice and does not model informative missingness.",
            "- Site is deliberately excluded, so center heterogeneity is not modeled; site sensitivity analysis remains future work.",
            "- No external validation is performed.",
            "- No costs, acquisition order, STOP rule, or patient-specific feature selection are evaluated.",
            "",
        ]
    )
    return "\n".join(lines)


def run_baseline(
    *,
    project_root: Path,
    cohort_path: Path,
    cohort_manifest_path: Path,
    model_path: Path,
    metrics_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    splits = load_cohort_splits(cohort_path)
    fitted = fit_cox_baseline(splits)
    metrics = evaluate_splits(splits, fitted)
    results = build_results(
        project_root=project_root,
        cohort_path=cohort_path,
        cohort_manifest_path=cohort_manifest_path,
        fitted=fitted,
        metrics=metrics,
    )
    save_model_artifact(fitted, model_path)
    write_results_json(results, metrics_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_markdown_report(results), encoding="utf-8")
    return results
