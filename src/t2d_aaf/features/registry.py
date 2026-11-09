from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from t2d_aaf.data.cohort import (
    BASE_FEATURES,
    CORE_ACQUIRABLE_FEATURES,
    EXCLUDED_RAW_FIELDS,
    METADATA_COLUMNS,
    OPTIONAL_FEATURES,
    TARGET_COLUMNS,
)


FEATURE_REGISTRY_SCHEMA_VERSION = "feature_registry_v1"
COST_CONFIG_SCHEMA_VERSION = "relative_costs_v1"
ROLE_NAMES: tuple[str, ...] = ("base", "acquirable", "optional", "metadata", "target")

EXPECTED_ROLE_FEATURES: dict[str, tuple[str, ...]] = {
    "base": BASE_FEATURES,
    "acquirable": CORE_ACQUIRABLE_FEATURES,
    "optional": OPTIONAL_FEATURES,
    "metadata": METADATA_COLUMNS,
    "target": TARGET_COLUMNS,
}


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    role: str
    raw_source: str
    clinical_group: str
    value_type: str
    unit: str
    rationale: str


@dataclass(frozen=True)
class RawExcludedField:
    name: str
    mapped_to: str | None
    reason: str


@dataclass(frozen=True)
class FeatureRegistry:
    schema_version: str
    dataset_schema_version: str
    description: str
    features: tuple[FeatureDefinition, ...]
    raw_excluded_fields: tuple[RawExcludedField, ...]

    def names_for_role(self, role: str) -> tuple[str, ...]:
        if role not in ROLE_NAMES:
            raise KeyError(f"Unknown feature role: {role}")
        return tuple(feature.name for feature in self.features if feature.role == role)

    @property
    def cohort_feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    def get(self, name: str) -> FeatureDefinition:
        for feature in self.features:
            if feature.name == name:
                return feature
        raise KeyError(f"Unknown cohort feature: {name}")


@dataclass(frozen=True)
class CostScheme:
    name: str
    role: str
    description: str
    costs: Mapping[str, float]


@dataclass(frozen=True)
class CostConfig:
    schema_version: str
    feature_registry_schema_version: str
    default_scheme: str
    semantics: Mapping[str, Any]
    base_feature_costs: Mapping[str, float]
    schemes: Mapping[str, CostScheme]

    @property
    def stop_cost(self) -> float:
        return float(self.semantics["stop_cost"])

    def scheme(self, name: str | None = None) -> CostScheme:
        selected = self.default_scheme if name is None else name
        try:
            return self.schemes[selected]
        except KeyError as exc:
            raise KeyError(f"Unknown cost scheme: {selected}") from exc

    def acquisition_cost(self, features: Iterable[str], *, scheme: str | None = None) -> float:
        requested = tuple(features)
        if len(requested) != len(set(requested)):
            raise ValueError("An acquirable feature cannot be charged more than once in one path.")
        selected = self.scheme(scheme)
        unknown = sorted(set(requested) - set(selected.costs))
        if unknown:
            raise KeyError(f"Features are not acquirable under {selected.name}: {unknown}")
        return float(sum(selected.costs[feature] for feature in requested))


@dataclass(frozen=True)
class FeatureCostDesign:
    registry: FeatureRegistry
    costs: CostConfig

    def validate_cohort_columns(self, columns: Iterable[str]) -> None:
        observed = tuple(columns)
        if len(observed) != len(set(observed)):
            raise ValueError("Cohort columns must be unique.")
        expected = set(self.registry.cohort_feature_names)
        actual = set(observed)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(f"Cohort/config mismatch; missing={missing}, unexpected={unexpected}")

    def validate_cohort_manifest(self, manifest: Mapping[str, Any]) -> None:
        if manifest.get("schema_version") != self.registry.dataset_schema_version:
            raise ValueError("cohort construction manifest schema differs from the feature registry dataset schema.")
        comparisons = {
            "base_features": self.registry.names_for_role("base"),
            "core_acquirable_features": self.registry.names_for_role("acquirable"),
            "optional_features": self.registry.names_for_role("optional"),
            "metadata_columns": self.registry.names_for_role("metadata"),
            "target_columns": self.registry.names_for_role("target"),
        }
        for key, expected in comparisons.items():
            observed = tuple(manifest.get(key, ()))
            if set(observed) != set(expected):
                raise ValueError(f"cohort construction manifest field {key!r} differs from feature and cost design config.")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        document = yaml.safe_load(file)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return document


def _required_text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be non-empty text.")
    return value.strip()


def _finite_cost(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric.")
    cost = float(value)
    if not math.isfinite(cost):
        raise ValueError(f"{context} must be finite.")
    return cost


def load_feature_registry(path: Path) -> FeatureRegistry:
    document = _load_yaml_mapping(path)
    schema_version = _required_text(document, "schema_version", "registry")
    if schema_version != FEATURE_REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported feature registry schema: {schema_version}")
    roles = document.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(ROLE_NAMES):
        raise ValueError(f"Feature roles must be exactly {ROLE_NAMES}.")

    features: list[FeatureDefinition] = []
    for role in ROLE_NAMES:
        entries = roles[role]
        if not isinstance(entries, list):
            raise ValueError(f"Role {role!r} must contain a list.")
        for index, entry in enumerate(entries):
            context = f"roles.{role}[{index}]"
            if not isinstance(entry, dict):
                raise ValueError(f"{context} must be a mapping.")
            features.append(
                FeatureDefinition(
                    name=_required_text(entry, "name", context),
                    role=role,
                    raw_source=_required_text(entry, "raw_source", context),
                    clinical_group=_required_text(entry, "clinical_group", context),
                    value_type=_required_text(entry, "value_type", context),
                    unit=_required_text(entry, "unit", context),
                    rationale=_required_text(entry, "rationale", context),
                )
            )

    names = [feature.name for feature in features]
    if len(names) != len(set(names)):
        raise ValueError("Every processed cohort feature must have exactly one role.")
    for role, expected in EXPECTED_ROLE_FEATURES.items():
        observed = tuple(feature.name for feature in features if feature.role == role)
        if observed != expected:
            raise ValueError(f"Role {role!r} must match the confirmed cohort construction order {expected}.")

    excluded_entries = document.get("raw_excluded_fields")
    if not isinstance(excluded_entries, list):
        raise ValueError("raw_excluded_fields must be a list.")
    excluded: list[RawExcludedField] = []
    for index, entry in enumerate(excluded_entries):
        context = f"raw_excluded_fields[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{context} must be a mapping.")
        mapped_to = entry.get("mapped_to")
        if mapped_to is not None and (not isinstance(mapped_to, str) or not mapped_to.strip()):
            raise ValueError(f"{context}.mapped_to must be text or null.")
        excluded.append(
            RawExcludedField(
                name=_required_text(entry, "name", context),
                mapped_to=mapped_to.strip() if isinstance(mapped_to, str) else None,
                reason=_required_text(entry, "reason", context),
            )
        )
    if {item.name for item in excluded} != set(EXCLUDED_RAW_FIELDS):
        raise ValueError("Raw excluded fields must match the confirmed cohort construction leakage contract.")
    valid_mappings = set(names)
    for item in excluded:
        if item.mapped_to is not None and item.mapped_to not in valid_mappings:
            raise ValueError(f"Excluded field {item.name!r} maps to unknown feature {item.mapped_to!r}.")

    return FeatureRegistry(
        schema_version=schema_version,
        dataset_schema_version=_required_text(document, "dataset_schema_version", "registry"),
        description=_required_text(document, "description", "registry"),
        features=tuple(features),
        raw_excluded_fields=tuple(excluded),
    )


def load_cost_config(path: Path, registry: FeatureRegistry) -> CostConfig:
    document = _load_yaml_mapping(path)
    schema_version = _required_text(document, "schema_version", "costs")
    if schema_version != COST_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported cost config schema: {schema_version}")
    registry_schema = _required_text(document, "feature_registry_schema_version", "costs")
    if registry_schema != registry.schema_version:
        raise ValueError("Cost config references a different feature registry schema.")

    semantics = document.get("semantics")
    if not isinstance(semantics, dict):
        raise ValueError("costs.semantics must be a mapping.")
    if semantics.get("unit_type") != "relative_experimental":
        raise ValueError("Cost units must be labeled relative_experimental.")
    if semantics.get("currency") is not None or semantics.get("real_prices") is not False:
        raise ValueError("feature and cost design costs must not claim a currency or real prices.")
    if semantics.get("base_features_are_zero_incremental_cost") is not True:
        raise ValueError("feature and cost design must explicitly mark base features as zero incremental cost.")
    if semantics.get("shared_panel_or_blood_draw_discount") is not False:
        raise ValueError("Shared panel or blood-draw discounts are not supported in feature and cost design.")
    if semantics.get("missing_result_policy") != "deferred_to_acquisition":
        raise ValueError("Missing-result behavior must remain deferred in feature and cost design.")
    stop_cost = _finite_cost(semantics.get("stop_cost"), "semantics.stop_cost")
    if stop_cost != 0.0:
        raise ValueError("STOP must have zero cost.")
    if semantics.get("aggregation") != "additive_per_feature":
        raise ValueError("feature and cost design supports additive_per_feature aggregation only.")

    base_document = document.get("base_feature_costs")
    if not isinstance(base_document, dict) or set(base_document) != set(BASE_FEATURES):
        raise ValueError("Base cost keys must match the confirmed base feature set.")
    base_costs = {
        name: _finite_cost(value, f"base_feature_costs.{name}")
        for name, value in base_document.items()
    }
    if any(cost != 0.0 for cost in base_costs.values()):
        raise ValueError("All feature and cost design base features must have zero incremental cost.")

    schemes_document = document.get("schemes")
    if not isinstance(schemes_document, dict) or not schemes_document:
        raise ValueError("At least one cost scheme is required.")
    candidate_names = set(registry.names_for_role("acquirable"))
    schemes: dict[str, CostScheme] = {}
    for name, entry in schemes_document.items():
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            raise ValueError("Every cost scheme must have a non-empty name and mapping body.")
        cost_document = entry.get("costs")
        if not isinstance(cost_document, dict) or set(cost_document) != candidate_names:
            raise ValueError(f"Scheme {name!r} must cost every and only core acquirable feature.")
        scheme_costs = {
            feature: _finite_cost(value, f"schemes.{name}.costs.{feature}")
            for feature, value in cost_document.items()
        }
        if any(cost <= 0.0 for cost in scheme_costs.values()):
            raise ValueError(f"Scheme {name!r} must use strictly positive acquisition costs.")
        schemes[name] = CostScheme(
            name=name,
            role=_required_text(entry, "role", f"schemes.{name}"),
            description=_required_text(entry, "description", f"schemes.{name}"),
            costs=scheme_costs,
        )

    default_scheme = _required_text(document, "default_scheme", "costs")
    if default_scheme not in schemes:
        raise ValueError("default_scheme does not exist in schemes.")
    if schemes[default_scheme].role != "primary":
        raise ValueError("The default cost scheme must have role 'primary'.")
    if sum(scheme.role == "primary" for scheme in schemes.values()) != 1:
        raise ValueError("Exactly one primary cost scheme is required.")
    if any(
        scheme.role != "sensitivity_only"
        for name, scheme in schemes.items()
        if name != default_scheme
    ):
        raise ValueError("Every non-default cost scheme must have role 'sensitivity_only'.")

    return CostConfig(
        schema_version=schema_version,
        feature_registry_schema_version=registry_schema,
        default_scheme=default_scheme,
        semantics=semantics,
        base_feature_costs=base_costs,
        schemes=schemes,
    )


def load_feature_cost_design(feature_path: Path, cost_path: Path) -> FeatureCostDesign:
    registry = load_feature_registry(feature_path)
    costs = load_cost_config(cost_path, registry)
    return FeatureCostDesign(registry=registry, costs=costs)
