from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from t2d_aaf.acquisition.adaptive import (
    STOP_INDEX,
    LockedAdaptivePolicy,
    PreparedAdaptiveData,
    load_adaptive_model_bundle,
    load_locked_adaptive_policy,
    prepare_adaptive_data,
    recommend_adaptive_actions,
    subset_to_mask,
)
from t2d_aaf.data.cohort import BASE_FEATURES, CORE_ACQUIRABLE_FEATURES
from t2d_aaf.features.registry import CostConfig, load_feature_cost_design
from t2d_aaf.survival.baseline import PREDICTOR_FEATURES, load_cohort_splits


FEATURE_LABELS: Mapping[str, str] = {
    "age": "Age",
    "gender": "Sex code",
    "bmi": "BMI",
    "sbp": "Systolic blood pressure",
    "dbp": "Diastolic blood pressure",
    "family_history_diabetes": "Family history of diabetes",
    "fpg": "Fasting plasma glucose",
    "total_cholesterol": "Total cholesterol",
    "triglycerides": "Triglycerides",
    "alt": "ALT",
    "ccr": "CCR",
    "bun": "BUN",
}

FEATURE_UNITS: Mapping[str, str] = {
    "age": "years",
    "gender": "1 male, 2 female",
    "bmi": "kg/m2",
    "sbp": "mmHg",
    "dbp": "mmHg",
    "family_history_diabetes": "1 yes, 0 no",
    "fpg": "mmol/L",
    "total_cholesterol": "mmol/L",
    "triglycerides": "mmol/L",
    "alt": "U/L",
    "ccr": "umol/L as labeled",
    "bun": "mmol/L",
}


@dataclass(frozen=True)
class DisplayValue:
    feature: str
    label: str
    value: float | None
    unit: str
    source_observed: bool


@dataclass(frozen=True)
class RiskTrajectoryPoint:
    step: int
    state_label: str
    risk_at_3_years: float
    cumulative_cost: float


@dataclass(frozen=True)
class DemoSnapshot:
    case_number: int
    case_label: str
    base_values: tuple[DisplayValue, ...]
    acquired_values: tuple[DisplayValue, ...]
    acquired_features: tuple[str, ...]
    risk_trajectory: tuple[RiskTrajectoryPoint, ...]
    current_risk_at_3_years: float
    total_cost: float
    next_action: str | None
    next_action_cost: float
    next_action_predicted_benefit: float | None
    next_action_utility: float | None

    @property
    def stopped(self) -> bool:
        return self.next_action is None


@dataclass(frozen=True)
class DemoService:
    predictor_frame: pd.DataFrame
    prepared: PreparedAdaptiveData
    policy: LockedAdaptivePolicy
    costs: CostConfig

    def __post_init__(self) -> None:
        if tuple(self.predictor_frame.columns) != tuple(PREDICTOR_FEATURES):
            raise ValueError("Demo frame must contain only the locked predictor columns.")
        if len(self.predictor_frame) != self.prepared.participants:
            raise ValueError("Demo frame and prepared policy data differ in size.")
        if self.predictor_frame.empty:
            raise ValueError("Demo requires at least one retrospective test participant.")

    @classmethod
    def from_project(cls, project_root: Path) -> "DemoService":

        cohort_path = project_root / "data" / "processed" / "primary_cohort.csv"
        static_metrics_path = project_root / "reports" / "static_baselines_metrics.json"
        static_model_path = project_root / "models" / "static_subsets.joblib"
        adaptive_model_path = project_root / "models" / "adaptive_value_policy.joblib"
        required_paths = (
            cohort_path,
            static_metrics_path,
            static_model_path,
            adaptive_model_path,
            project_root / "configs" / "features.yaml",
            project_root / "configs" / "costs.yaml",
        )
        missing = [path for path in required_paths if not path.exists()]
        if missing:
            relative = [path.relative_to(project_root).as_posix() for path in missing]
            raise FileNotFoundError(f"Missing local cohort or model artifacts: {relative}")

        splits = load_cohort_splits(cohort_path)
        predictor_frame = splits.test.loc[:, PREDICTOR_FEATURES].copy()
        design = load_feature_cost_design(
            project_root / "configs" / "features.yaml",
            project_root / "configs" / "costs.yaml",
        )
        model_bundle = load_adaptive_model_bundle(
            static_metrics_path,
            static_model_path,
        )
        policy = load_locked_adaptive_policy(adaptive_model_path)
        prepared = prepare_adaptive_data(model_bundle, predictor_frame)
        return cls(
            predictor_frame=predictor_frame,
            prepared=prepared,
            policy=policy,
            costs=design.costs,
        )

    @property
    def participant_count(self) -> int:
        return len(self.predictor_frame)

    def snapshot(
        self,
        case_number: int,
        acquired_features: Sequence[str] = (),
    ) -> DemoSnapshot:

        if isinstance(case_number, bool) or not 1 <= case_number <= self.participant_count:
            raise ValueError(
                f"case_number must be between 1 and {self.participant_count}."
            )
        acquired = tuple(acquired_features)
        if len(acquired) != len(set(acquired)):
            raise ValueError("A demo path cannot acquire the same feature twice.")
        unknown = sorted(set(acquired) - set(CORE_ACQUIRABLE_FEATURES))
        if unknown:
            raise ValueError(f"Unknown acquired features: {unknown}")

        row_index = case_number - 1
        mask = subset_to_mask(acquired)
        recommendation = recommend_adaptive_actions(
            self.prepared,
            self.policy.value_models,
            self.costs,
            self.policy.selected_cost_penalty,
            np.asarray([mask], dtype=np.int64),
            rows=np.asarray([row_index], dtype=np.int64),
        )
        action_index = int(recommendation.action_indices[0])
        next_action = (
            None if action_index == STOP_INDEX else CORE_ACQUIRABLE_FEATURES[action_index]
        )
        next_cost = (
            self.costs.stop_cost
            if next_action is None
            else float(self.costs.scheme().costs[next_action])
        )
        next_benefit = (
            None
            if next_action is None
            else float(recommendation.predicted_benefits[0, action_index])
        )
        next_utility = (
            None
            if next_action is None
            else float(recommendation.utilities[0, action_index])
        )
        row = self.predictor_frame.iloc[row_index]
        trajectory = self._risk_trajectory(row_index, acquired)
        return DemoSnapshot(
            case_number=case_number,
            case_label=f"Test case {case_number:05d}",
            base_values=tuple(self._display_value(row, feature) for feature in BASE_FEATURES),
            acquired_values=tuple(
                self._display_value(row, feature) for feature in acquired
            ),
            acquired_features=acquired,
            risk_trajectory=trajectory,
            current_risk_at_3_years=float(trajectory[-1].risk_at_3_years),
            total_cost=float(trajectory[-1].cumulative_cost),
            next_action=next_action,
            next_action_cost=next_cost,
            next_action_predicted_benefit=next_benefit,
            next_action_utility=next_utility,
        )

    def acquire_recommended(self, snapshot: DemoSnapshot) -> tuple[str, ...]:

        if snapshot.next_action is None:
            return snapshot.acquired_features
        return (*snapshot.acquired_features, snapshot.next_action)

    def _risk_trajectory(
        self,
        row_index: int,
        acquired: Sequence[str],
    ) -> tuple[RiskTrajectoryPoint, ...]:
        points: list[RiskTrajectoryPoint] = []
        prefix: list[str] = []
        for step in range(len(acquired) + 1):
            mask = subset_to_mask(prefix)
            points.append(
                RiskTrajectoryPoint(
                    step=step,
                    state_label="Base" if not prefix else FEATURE_LABELS[prefix[-1]],
                    risk_at_3_years=float(self.prepared.subset_risks[row_index, mask]),
                    cumulative_cost=self.costs.acquisition_cost(prefix),
                )
            )
            if step < len(acquired):
                prefix.append(acquired[step])
        return tuple(points)

    @staticmethod
    def _display_value(row: pd.Series, feature: str) -> DisplayValue:
        raw_value = row[feature]
        observed = not pd.isna(raw_value)
        return DisplayValue(
            feature=feature,
            label=FEATURE_LABELS[feature],
            value=float(raw_value) if observed else None,
            unit=FEATURE_UNITS[feature],
            source_observed=observed,
        )


def deterministic_case_number(seed: int, participant_count: int) -> int:

    if participant_count <= 0:
        raise ValueError("participant_count must be positive.")
    rng = np.random.default_rng(seed)
    return int(rng.integers(1, participant_count + 1))


def predictor_columns_for_demo() -> tuple[str, ...]:

    return tuple(PREDICTOR_FEATURES)
