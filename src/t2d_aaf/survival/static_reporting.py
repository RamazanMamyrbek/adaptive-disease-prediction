from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from t2d_aaf.data.cohort import (
    BASE_FEATURES,
    CORE_ACQUIRABLE_FEATURES,
    DEFAULT_SEED,
    EXCLUDED_FROM_MAIN_PREDICTORS,
    OPTIONAL_FEATURES,
)
from t2d_aaf.features.registry import CostConfig, load_feature_cost_design
from t2d_aaf.survival.baseline import (
    EVALUATION_HORIZON_YEARS,
    MODEL_TYPE,
    TIES_METHOD,
    load_cohort_splits,
    preprocessing_summary,
)
from t2d_aaf.survival.reporting import package_versions, portable_path, sha256_file
from t2d_aaf.survival.static_baselines import (
    DEFAULT_RANDOM_REPEATS,
    EVALUATION_SPLITS,
    STATIC_RANDOM_SEED,
    OPTIMIZER,
    StaticSubsetRun,
    canonical_subset,
    derive_static_orders,
    fit_all_static_subsets,
    generate_random_orders,
    prefix_subsets,
    subset_id,
)


SUMMARY_METRICS: tuple[str, ...] = (
    "harrell_c_index",
    "ipcw_brier_at_horizon",
    "mean_predicted_risk_at_horizon",
    "calibration_in_the_large_difference",
)


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _subset_record(
    subset: tuple[str, ...],
    run: StaticSubsetRun,
    costs: CostConfig,
) -> dict[str, Any]:
    fitted = run.fitted_models[subset]
    return {
        "subset_id": subset_id(subset),
        "acquired_features": list(subset),
        "acquired_count": len(subset),
        "predictor_features": list(fitted.selected_feature_names),
        "cost_by_scheme": {
            name: costs.acquisition_cost(subset, scheme=name) for name in costs.schemes
        },
        "fit_warnings": list(fitted.fit_warnings),
        "metrics": run.evaluations[subset],
    }


def _ordered_path(
    order: Sequence[str],
    run: StaticSubsetRun,
    costs: CostConfig,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for acquired_count, subset in enumerate(prefix_subsets(order)):
        point = _subset_record(subset, run, costs)
        point["acquired_count"] = acquired_count
        point["added_feature"] = None if acquired_count == 0 else order[acquired_count - 1]
        points.append(point)
    return points


def _random_baseline(
    run: StaticSubsetRun,
    costs: CostConfig,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    orders = generate_random_orders(repeats=repeats, seed=seed)
    points: list[dict[str, Any]] = []
    for acquired_count in range(len(CORE_ACQUIRABLE_FEATURES) + 1):
        subsets = [canonical_subset(order[:acquired_count]) for order in orders]
        point: dict[str, Any] = {
            "acquired_count": acquired_count,
            "unique_subsets_sampled": len(set(subsets)),
            "cost_by_scheme": {
                scheme_name: _metric_summary(
                    [costs.acquisition_cost(subset, scheme=scheme_name) for subset in subsets]
                )
                for scheme_name in costs.schemes
            },
            "metrics": {},
        }
        for split in EVALUATION_SPLITS:
            point["metrics"][split] = {
                metric: _metric_summary(
                    [float(run.evaluations[subset][split][metric]) for subset in subsets]
                )
                for metric in SUMMARY_METRICS
            }
        points.append(point)
    return {
        "definition": "fixed-seed random permutations; each K uses the first K laboratories",
        "seed": seed,
        "repeats": repeats,
        "selection_data": "none",
        "points": points,
    }


def build_static_results(
    *,
    project_root: Path,
    cohort_path: Path,
    cohort_manifest_path: Path,
    feature_config_path: Path,
    cost_config_path: Path,
    run: StaticSubsetRun,
    random_repeats: int,
    random_seed: int,
) -> dict[str, Any]:
    design = load_feature_cost_design(feature_config_path, cost_config_path)
    orders = derive_static_orders(run.evaluations)
    base_subset: tuple[str, ...] = ()
    full_subset = tuple(CORE_ACQUIRABLE_FEATURES)
    subset_records = [
        _subset_record(subset, run, design.costs)
        for subset in sorted(run.evaluations, key=lambda item: (len(item), item))
    ]
    return {
        "schema_version": "static_baselines_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 5,
        "scope": "global static feature-set baselines; no ordered NEXT/STOP or adaptive policy",
        "seeds": {
            "cohort_split": DEFAULT_SEED,
            "random_baseline": random_seed,
        },
        "dataset": {
            "cohort_path": portable_path(cohort_path, project_root),
            "cohort_sha256": sha256_file(cohort_path),
            "cohort_manifest_path": portable_path(cohort_manifest_path, project_root),
            "cohort_manifest_sha256": sha256_file(cohort_manifest_path),
        },
        "target": {"event": "event", "duration": "duration_years", "censoring": "event = 0"},
        "features": {
            "base": list(BASE_FEATURES),
            "acquisition_candidates": list(CORE_ACQUIRABLE_FEATURES),
            "optional_not_used": list(OPTIONAL_FEATURES),
            "excluded_from_predictors": list(EXCLUDED_FROM_MAIN_PREDICTORS),
        },
        "model": {
            "type": MODEL_TYPE,
            "ties": TIES_METHOD,
            "penalization": None,
            "optimizer": OPTIMIZER,
            "fit_split": "train",
            "models_fitted": len(run.fitted_models),
            "all_candidate_subsets_fitted": True,
        },
        "preprocessing": preprocessing_summary(run.preprocessor),
        "evaluation": {
            "horizon_years": EVALUATION_HORIZON_YEARS,
            "selection_metric": "validation Harrell C-index",
            "test_used_for_selection": False,
            "reported_splits": list(EVALUATION_SPLITS),
            "brier": "single-horizon IPCW; censoring distribution estimated on train only",
        },
        "costs": {
            "config_path": portable_path(cost_config_path, project_root),
            "default_scheme": design.costs.default_scheme,
            "unit_type": design.costs.semantics["unit_type"],
            "real_prices": design.costs.semantics["real_prices"],
            "schemes": {
                name: {
                    "role": scheme.role,
                    "costs": dict(scheme.costs),
                }
                for name, scheme in design.costs.schemes.items()
            },
        },
        "methods": {
            "base_only": {
                "definition": "all base features and no laboratory candidates",
                "selection_data": "none",
                "result": _subset_record(base_subset, run, design.costs),
            },
            "full_features": {
                "definition": "all base features and all six laboratory candidates",
                "selection_data": "none",
                "result": _subset_record(full_subset, run, design.costs),
            },
            "static_top_k": {
                "definition": "fixed order ranked by singleton incremental validation C-index over base",
                "selection_data": "validation only",
                "order": list(orders["static_top_k"]),
                "points": _ordered_path(orders["static_top_k"], run, design.costs),
            },
            "random": _random_baseline(
                run,
                design.costs,
                repeats=random_repeats,
                seed=random_seed,
            ),
            "importance_greedy": {
                "definition": "fixed global forward order maximizing validation C-index at each step",
                "selection_data": "validation only",
                "order": list(orders["importance_greedy"]),
                "points": _ordered_path(orders["importance_greedy"], run, design.costs),
            },
        },
        "all_subset_results": subset_records,
        "acquisition_policy": None,
        "environment": package_versions(),
    }


def save_static_artifact(run: StaticSubsetRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": "static_subset_models_v1",
            "preprocessor": run.preprocessor,
            "transformed_feature_names": run.transformed_feature_names,
            "models": dict(run.fitted_models),
        },
        output_path,
    )


def write_results_json(results: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def _metric(value: float) -> str:
    return f"{value:.4f}"


def _ordered_table(lines: list[str], method: Mapping[str, Any]) -> None:
    lines.extend(
        [
            "| K | Added | Acquired subset | Cost | Validation C-index | Test C-index | Validation Brier @ 3y | Test Brier @ 3y |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for point in method["points"]:
        subset = ", ".join(point["acquired_features"]) or "none"
        added = point["added_feature"] or "-"
        validation = point["metrics"]["validation"]
        test = point["metrics"]["test"]
        lines.append(
            f"| {point['acquired_count']} | `{added}` | {subset} | "
            f"{point['cost_by_scheme']['uniform_unit']:.1f} | "
            f"{_metric(validation['harrell_c_index'])} | {_metric(test['harrell_c_index'])} | "
            f"{_metric(validation['ipcw_brier_at_horizon'])} | "
            f"{_metric(test['ipcw_brier_at_horizon'])} |"
        )


def build_markdown_report(results: Mapping[str, Any]) -> str:
    methods = results["methods"]
    base = methods["base_only"]["result"]
    full = methods["full_features"]["result"]
    lines = [
        "# static baseline Static Baselines",
        "",
        "static baseline compares fixed, population-level laboratory feature sets. It does not implement patient-specific acquisition, an ordered `NEXT/STOP` process, or an adaptive policy.",
        "",
        "## Experiment Contract",
        "",
        f"- Cohort: `{results['dataset']['cohort_path']}`",
        f"- Cohort SHA-256: `{results['dataset']['cohort_sha256']}`",
        "- Event/duration: `event`, `duration_years`; `event = 0` is right-censoring.",
        "- Model: unpenalized Cox PH with Breslow ties and BFGS optimization for every feature subset.",
        "- Preprocessing: median/mode imputation and continuous scaling fitted once on train only.",
        "- Selection: Top-K and greedy orders use validation C-index only.",
        "- Test: evaluated only after the fixed orders are derived; never used to choose a feature or order.",
        f"- Candidate subsets fitted: {results['model']['models_fitted']} of 64.",
        f"- Primary cost scheme: `{results['costs']['default_scheme']}` relative units; no real prices are claimed.",
        "",
        "## Features And Leakage Control",
        "",
        "Base features: " + ", ".join(f"`{item}`" for item in results["features"]["base"]),
        "",
        "Laboratory candidates: "
        + ", ".join(f"`{item}`" for item in results["features"]["acquisition_candidates"]),
        "",
        "Excluded predictors remain `participant_id`, `site`, `event`, `duration_years`, final-visit FPG, and the follow-up diagnosis flag. Optional cohort construction fields are not used. No participant-level predictions are written to the repository.",
        "",
        "## Endpoints",
        "",
        "| Method | Labs | Cost | Validation C-index | Test C-index | Validation Brier @ 3y | Test Brier @ 3y |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, record in (("Base Only", base), ("Full Features", full)):
        validation = record["metrics"]["validation"]
        test = record["metrics"]["test"]
        lines.append(
            f"| {label} | {record['acquired_count']} | "
            f"{record['cost_by_scheme']['uniform_unit']:.1f} | "
            f"{_metric(validation['harrell_c_index'])} | {_metric(test['harrell_c_index'])} | "
            f"{_metric(validation['ipcw_brier_at_horizon'])} | "
            f"{_metric(test['ipcw_brier_at_horizon'])} |"
        )

    top_k = methods["static_top_k"]
    lines.extend(
        [
            "",
            "## Static Top-K",
            "",
            "Candidates are ranked once by the validation C-index of `base + one candidate`; prefixes of that fixed order define K=0..6.",
            "",
            "Order: " + " -> ".join(f"`{item}`" for item in top_k["order"]),
            "",
        ]
    )
    _ordered_table(lines, top_k)

    greedy = methods["importance_greedy"]
    lines.extend(
        [
            "",
            "## Global Importance Greedy",
            "",
            "Starting from base only, each step adds the remaining candidate with the highest validation C-index for the resulting subset. The resulting order is global and fixed for every participant.",
            "",
            "Order: " + " -> ".join(f"`{item}`" for item in greedy["order"]),
            "",
        ]
    )
    _ordered_table(lines, greedy)

    random_method = methods["random"]
    lines.extend(
        [
            "",
            "## Random Acquisition",
            "",
            f"The random reference uses {random_method['repeats']:,} fixed-seed permutations (`seed={random_method['seed']}`). At each K, metrics are summarized across the corresponding random prefix subsets; no data select the order.",
            "",
            "| K | Unique subsets | Mean cost | Validation C-index mean (SD) | Test C-index mean (SD) | Validation Brier mean | Test Brier mean |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for point in random_method["points"]:
        validation = point["metrics"]["validation"]
        test = point["metrics"]["test"]
        lines.append(
            f"| {point['acquired_count']} | {point['unique_subsets_sampled']} | "
            f"{point['cost_by_scheme']['uniform_unit']['mean']:.1f} | "
            f"{_metric(validation['harrell_c_index']['mean'])} ({_metric(validation['harrell_c_index']['sd'])}) | "
            f"{_metric(test['harrell_c_index']['mean'])} ({_metric(test['harrell_c_index']['sd'])}) | "
            f"{_metric(validation['ipcw_brier_at_horizon']['mean'])} | "
            f"{_metric(test['ipcw_brier_at_horizon']['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These results quantify the cost-performance frontier available to fixed population-level feature sets. They are comparators for later acquisition stages, not an Active Feature Acquisition result and not a clinical deployment claim.",
            "",
            f"The base-only model reaches validation/test C-indices of {_metric(base['metrics']['validation']['harrell_c_index'])}/{_metric(base['metrics']['test']['harrell_c_index'])}. Adding FPG alone raises them to {_metric(greedy['points'][1]['metrics']['validation']['harrell_c_index'])}/{_metric(greedy['points'][1]['metrics']['test']['harrell_c_index'])} at one relative cost unit, capturing most of the discrimination available from the six-laboratory full model ({_metric(full['metrics']['validation']['harrell_c_index'])}/{_metric(full['metrics']['test']['harrell_c_index'])} at six units). The validation-selected greedy path improves monotonically through all six steps; its five-feature prefix has the highest reported test C-index ({_metric(greedy['points'][5]['metrics']['test']['harrell_c_index'])}), but this test observation did not select the prefix and must not be used to redesign the order.",
            "",
            "No participant or clinical value was excluded in static baseline. All cohort construction cohort members remain in their original participant-level train/validation/test splits.",
            "",
            "## Limitations",
            "",
            "- Orders are selected on one validation split without resampling or uncertainty intervals.",
            "- All subset results share the simple baseline modelling median/mode preprocessing and unpenalized Cox specification.",
            "- BFGS is used uniformly because the statsmodels default Newton optimizer produced non-finite coefficients for `base + alt`; no ALT values were removed or clipped.",
            "- The proportional-hazards assumption is not formally tested here.",
            "- The primary cost scheme counts tests equally and does not represent prices, panels, shared blood draws, or patient burden.",
            "- Missing laboratory results are imputed after an action is treated as acquired; missing-result acquisition behavior is deferred to the acquisition stages.",
            "- The test split is reported for many static subsets. Future method design must remain locked to validation results to avoid test-set feedback.",
            "- There is no external validation or site sensitivity analysis.",
            "",
        ]
    )
    return "\n".join(lines)


def run_static_baselines(
    *,
    project_root: Path,
    cohort_path: Path,
    cohort_manifest_path: Path,
    feature_config_path: Path,
    cost_config_path: Path,
    model_path: Path,
    metrics_path: Path,
    report_path: Path,
    random_repeats: int = DEFAULT_RANDOM_REPEATS,
    random_seed: int = STATIC_RANDOM_SEED,
) -> dict[str, Any]:
    if random_repeats <= 0:
        raise ValueError("random_repeats must be positive.")
    load_feature_cost_design(feature_config_path, cost_config_path)
    splits = load_cohort_splits(cohort_path)

    def show_progress(index: int, total: int, subset: tuple[str, ...]) -> None:
        label = ",".join(subset) if subset else "base_only"
        print(f"[{index:02d}/{total}] fitting {label}", flush=True)

    run = fit_all_static_subsets(splits, progress=show_progress)
    results = build_static_results(
        project_root=project_root,
        cohort_path=cohort_path,
        cohort_manifest_path=cohort_manifest_path,
        feature_config_path=feature_config_path,
        cost_config_path=cost_config_path,
        run=run,
        random_repeats=random_repeats,
        random_seed=random_seed,
    )
    save_static_artifact(run, model_path)
    write_results_json(results, metrics_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_markdown_report(results), encoding="utf-8")
    return results
