from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from t2d_aaf.features.registry import FeatureCostDesign, FeatureDefinition, load_feature_cost_design


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_cohort_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        try:
            return tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"Cohort CSV is empty: {path}") from exc


def build_feature_cost_manifest(
    *,
    project_root: Path,
    feature_config_path: Path,
    cost_config_path: Path,
    cohort_path: Path,
    cohort_manifest_path: Path,
    design: FeatureCostDesign,
) -> dict[str, Any]:
    cohort_manifest = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
    design.validate_cohort_columns(read_cohort_header(cohort_path))
    design.validate_cohort_manifest(cohort_manifest)

    registry = design.registry
    costs = design.costs
    return {
        "schema_version": "feature_cost_design_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 4,
        "scope": "feature roles and relative experimental costs; no model training or acquisition policy",
        "inputs": {
            "feature_config_path": portable_path(feature_config_path, project_root),
            "feature_config_sha256": sha256_file(feature_config_path),
            "cost_config_path": portable_path(cost_config_path, project_root),
            "cost_config_sha256": sha256_file(cost_config_path),
            "cohort_path": portable_path(cohort_path, project_root),
            "cohort_sha256": sha256_file(cohort_path),
            "cohort_manifest_path": portable_path(cohort_manifest_path, project_root),
            "cohort_manifest_sha256": sha256_file(cohort_manifest_path),
        },
        "feature_registry_schema_version": registry.schema_version,
        "cost_config_schema_version": costs.schema_version,
        "roles": {role: list(registry.names_for_role(role)) for role in ("base", "acquirable", "optional", "metadata", "target")},
        "raw_excluded_fields": [
            {"name": field.name, "mapped_to": field.mapped_to, "reason": field.reason}
            for field in registry.raw_excluded_fields
        ],
        "costs": {
            "default_scheme": costs.default_scheme,
            "semantics": dict(costs.semantics),
            "base_feature_costs": dict(costs.base_feature_costs),
            "schemes": {
                name: {
                    "role": scheme.role,
                    "description": scheme.description,
                    "costs": dict(scheme.costs),
                    "all_candidate_cost": costs.acquisition_cost(scheme.costs, scheme=name),
                }
                for name, scheme in costs.schemes.items()
            },
        },
        "validation": {
            "cohort_header_matches_registry": True,
            "cohort_manifest_matches_registry": True,
            "all_base_costs_zero": True,
            "all_candidate_costs_positive": True,
            "stop_cost_zero": True,
            "real_prices_claimed": False,
        },
        "model_training": None,
        "acquisition_policy": None,
    }


def _feature_table_rows(features: tuple[FeatureDefinition, ...]) -> list[str]:
    return [
        f"| {feature.name} | {feature.clinical_group} | {feature.value_type} | {feature.unit} | {feature.rationale} |"
        for feature in features
    ]


def build_markdown_report(manifest: dict[str, Any], design: FeatureCostDesign) -> str:
    registry = design.registry
    costs = design.costs
    lines = [
        "# feature and cost design Feature And Cost Design",
        "",
        "feature and cost design formalizes feature roles and relative experimental acquisition costs for later retrospective simulations. It does not train a model, run a static baseline, or implement an acquisition policy.",
        "",
        "## Validation Contract",
        "",
        f"- Feature registry: `{manifest['inputs']['feature_config_path']}`",
        f"- Cost configuration: `{manifest['inputs']['cost_config_path']}`",
        f"- Cohort: `{manifest['inputs']['cohort_path']}`",
        "- All 24 processed cohort columns have exactly one role.",
        "- Feature roles match the cohort construction manifest and code constants.",
        "- Costs are assigned only to the six core acquirable laboratory features.",
        "- No target, metadata, optional, identifier, or future field is acquirable.",
        "",
        "## Base Features",
        "",
        "Base features are treated as already available with zero incremental acquisition cost in the primary MVP.",
        "",
        "| Feature | Clinical group | Value type | Unit | Rationale |",
        "|---|---|---|---|---|",
        *_feature_table_rows(tuple(feature for feature in registry.features if feature.role == "base")),
        "",
        "This zero cost is a modeling boundary, not a claim that history, anthropometry, or blood-pressure measurement is free in real care. SBP and DBP may be moved to a low-cost acquirable tier in a later sensitivity analysis.",
        "",
        "## Core Acquirable Features",
        "",
        "Each core laboratory value is a separate acquisition action in version 1.",
        "",
        "| Feature | Clinical group | Value type | Unit | Rationale |",
        "|---|---|---|---|---|",
        *_feature_table_rows(tuple(feature for feature in registry.features if feature.role == "acquirable")),
        "",
        "## Relative Cost Schemes",
        "",
        "Costs are dimensionless experimental units. They are not prices, reimbursement amounts, or estimates of patient burden.",
        "",
        "| Feature | " + " | ".join(costs.schemes) + " |",
        "|---|" + "---:|" * len(costs.schemes),
    ]
    for feature in registry.names_for_role("acquirable"):
        lines.append(
            f"| {feature} | "
            + " | ".join(f"{costs.schemes[name].costs[feature]:.2f}" for name in costs.schemes)
            + " |"
        )
    lines.extend(
        [
            f"| **Acquire all six** | "
            + " | ".join(
                f"{costs.acquisition_cost(registry.names_for_role('acquirable'), scheme=name):.2f}"
                for name in costs.schemes
            )
            + " |",
            "",
            f"Default scheme: `{costs.default_scheme}`. It makes total cost equal to the number of acquired laboratory features. The two FPG schemes are sensitivity stress tests motivated by FPG's dominant baseline modelling prognostic contribution; they do not assert that FPG is cheaper or more expensive in practice.",
            "",
            "Cost aggregation is additive per feature. Version 1 does not model shared blood-draw costs, laboratory panels, bundled assays, repeated orders, or state-dependent discounts. `STOP` costs zero.",
            "",
            "## Optional Features",
            "",
            "Optional features remain in the processed cohort but are ineligible for the core acquisition action space.",
            "",
            "| Feature | Clinical group | Value type | Unit | Rationale |",
            "|---|---|---|---|---|",
            *_feature_table_rows(tuple(feature for feature in registry.features if feature.role == "optional")),
            "",
            "## Metadata And Targets",
            "",
            "| Feature | Role | Use |",
            "|---|---|---|",
        ]
    )
    for role in ("metadata", "target"):
        for feature in registry.features:
            if feature.role == role:
                lines.append(f"| {feature.name} | {role} | {feature.rationale} |")

    lines.extend(
        [
            "",
            "## Excluded Raw Fields",
            "",
            "| Raw field | Processed mapping | Reason |",
            "|---|---|---|",
        ]
    )
    for field in registry.raw_excluded_fields:
        lines.append(f"| {field.name} | {field.mapped_to or 'not retained'} | {field.reason} |")

    lines.extend(
        [
            "",
            "## Deferred Decisions",
            "",
            "- Participant-level handling of a requested laboratory value that is missing in the retrospective source data is deferred to the acquisition implementation stage.",
            "- No policy may silently use source missingness, future values, event status, or follow-up duration to choose an action.",
            "- Shared blood draw and panel economics require either justified external cost data or an explicit sensitivity model.",
            "- Real monetary or patient-burden costs are not available in the primary dataset.",
            "",
            "## feature and cost design Outcome",
            "",
            "The project now has a machine-validated action vocabulary and cost interface for the next static-baseline stage. No predictive result is produced or changed by feature and cost design.",
            "",
        ]
    )
    return "\n".join(lines)


def run_feature_cost_design(
    *,
    project_root: Path,
    feature_config_path: Path,
    cost_config_path: Path,
    cohort_path: Path,
    cohort_manifest_path: Path,
    output_manifest_path: Path,
    output_report_path: Path,
) -> dict[str, Any]:
    design = load_feature_cost_design(feature_config_path, cost_config_path)
    manifest = build_feature_cost_manifest(
        project_root=project_root,
        feature_config_path=feature_config_path,
        cost_config_path=cost_config_path,
        cohort_path=cohort_path,
        cohort_manifest_path=cohort_manifest_path,
        design=design,
    )
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(build_markdown_report(manifest, design), encoding="utf-8")
    return manifest
