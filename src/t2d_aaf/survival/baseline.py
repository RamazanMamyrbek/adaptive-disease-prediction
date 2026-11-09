from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from t2d_aaf.data.cohort import (
    BASE_FEATURES,
    CORE_ACQUIRABLE_FEATURES,
    EXCLUDED_FROM_MAIN_PREDICTORS,
)
from t2d_aaf.evaluation.survival_metrics import (
    harrell_c_index,
    ipcw_brier_score,
    kaplan_meier_curve,
    observed_event_risk_at_horizon,
)


MODEL_NAME = "cox_full_core_v1"
MODEL_TYPE = "Cox proportional hazards"
TIES_METHOD = "breslow"
EVALUATION_HORIZON_YEARS = 3.0

CONTINUOUS_FEATURES: tuple[str, ...] = (
    "age",
    "bmi",
    "sbp",
    "dbp",
    "fpg",
    "total_cholesterol",
    "triglycerides",
    "alt",
    "ccr",
    "bun",
)
BINARY_FEATURES: tuple[str, ...] = (
    "gender",
    "family_history_diabetes",
)
PREDICTOR_FEATURES: tuple[str, ...] = (*BASE_FEATURES, *CORE_ACQUIRABLE_FEATURES)
TARGET_COLUMNS: tuple[str, ...] = ("event", "duration_years")
METADATA_COLUMNS: tuple[str, ...] = ("participant_id", "split", "site")
SPLIT_ORDER: tuple[str, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class CohortSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True)
class FittedCoxBaseline:
    preprocessor: ColumnTransformer
    feature_names: tuple[str, ...]
    coefficients: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    p_values: NDArray[np.float64]
    baseline_event_times: NDArray[np.float64]
    baseline_cumulative_hazard: NDArray[np.float64]
    risk_center: float
    fit_warnings: tuple[str, ...]


def load_cohort_splits(cohort_path: Path) -> CohortSplits:

    frame = pd.read_csv(cohort_path, low_memory=False)
    required = set(PREDICTOR_FEATURES) | set(TARGET_COLUMNS) | set(METADATA_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cohort is missing required columns: {missing}")
    if frame["participant_id"].isna().any() or frame["participant_id"].duplicated().any():
        raise ValueError("participant_id must be non-missing and unique.")
    if set(frame["split"].unique()) != set(SPLIT_ORDER):
        raise ValueError(f"Expected splits {SPLIT_ORDER}; found {sorted(frame['split'].unique())}")
    if not frame["event"].isin((0, 1)).all():
        raise ValueError("event must contain only 0 and 1.")
    if frame["duration_years"].isna().any() or (frame["duration_years"] <= 0).any():
        raise ValueError("duration_years must be non-missing and positive.")

    forbidden_predictors = set(PREDICTOR_FEATURES) & set(EXCLUDED_FROM_MAIN_PREDICTORS)
    if forbidden_predictors:
        raise ValueError(f"Predictor contract includes excluded fields: {sorted(forbidden_predictors)}")
    for feature in PREDICTOR_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="raise")

    split_frames = {
        split: frame.loc[frame["split"] == split].reset_index(drop=True)
        for split in SPLIT_ORDER
    }
    return CohortSplits(
        train=split_frames["train"],
        validation=split_frames["validation"],
        test=split_frames["test"],
    )


def build_preprocessor() -> ColumnTransformer:

    continuous_pipeline = Pipeline(
        steps=(
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        )
    )
    binary_pipeline = Pipeline(
        steps=(("imputer", SimpleImputer(strategy="most_frequent")),)
    )
    return ColumnTransformer(
        transformers=(
            ("continuous", continuous_pipeline, list(CONTINUOUS_FEATURES)),
            ("binary", binary_pipeline, list(BINARY_FEATURES)),
        ),
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform_predictors(
    preprocessor: ColumnTransformer,
    frame: pd.DataFrame,
) -> NDArray[np.float64]:
    transformed = np.asarray(preprocessor.transform(frame.loc[:, PREDICTOR_FEATURES]), dtype=float)
    if transformed.ndim != 2 or not np.isfinite(transformed).all():
        raise ValueError("Preprocessing produced invalid predictor values.")
    return transformed


def breslow_baseline_cumulative_hazard(
    duration: NDArray[np.float64],
    event: NDArray[np.int64],
    linear_predictor: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:

    if not (len(duration) == len(event) == len(linear_predictor)):
        raise ValueError("Cox baseline vectors must have equal lengths.")
    risk_center = float(np.mean(linear_predictor))
    centered_predictor = linear_predictor - risk_center
    relative_hazard = np.exp(centered_predictor)
    if not np.isfinite(relative_hazard).all():
        raise ValueError("Cox relative hazards are non-finite.")

    order = np.argsort(-duration, kind="mergesort")
    sorted_duration = duration[order]
    sorted_event = event[order]
    cumulative_risk = np.cumsum(relative_hazard[order])

    event_times_descending: list[float] = []
    increments_descending: list[float] = []
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_duration[end] == sorted_duration[start]:
            end += 1
        event_count = int(sorted_event[start:end].sum())
        if event_count:
            risk_set_sum = float(cumulative_risk[end - 1])
            event_times_descending.append(float(sorted_duration[start]))
            increments_descending.append(event_count / risk_set_sum)
        start = end

    event_times = np.asarray(event_times_descending[::-1], dtype=float)
    cumulative_hazard = np.cumsum(np.asarray(increments_descending[::-1], dtype=float))
    return event_times, cumulative_hazard, risk_center


def fit_cox_baseline(splits: CohortSplits) -> FittedCoxBaseline:

    from statsmodels.duration.hazard_regression import PHReg

    preprocessor = build_preprocessor()
    train = splits.train
    train_matrix = np.asarray(
        preprocessor.fit_transform(train.loc[:, PREDICTOR_FEATURES]), dtype=float
    )
    if not np.isfinite(train_matrix).all():
        raise ValueError("Training preprocessing produced invalid predictor values.")

    duration = train["duration_years"].to_numpy(dtype=float)
    event = train["event"].to_numpy(dtype=np.int64)
    model = PHReg(duration, train_matrix, status=event, ties=TIES_METHOD, missing="raise")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(disp=False)

    coefficients = np.asarray(result.params, dtype=float)
    standard_errors = np.asarray(result.bse, dtype=float)
    p_values = np.asarray(result.pvalues, dtype=float)
    if not all(np.isfinite(values).all() for values in (coefficients, standard_errors, p_values)):
        raise ValueError("Cox fit returned non-finite parameter estimates.")

    linear_predictor = train_matrix @ coefficients
    event_times, cumulative_hazard, risk_center = breslow_baseline_cumulative_hazard(
        duration,
        event,
        linear_predictor,
    )
    return FittedCoxBaseline(
        preprocessor=preprocessor,
        feature_names=tuple(str(name) for name in preprocessor.get_feature_names_out()),
        coefficients=coefficients,
        standard_errors=standard_errors,
        p_values=p_values,
        baseline_event_times=event_times,
        baseline_cumulative_hazard=cumulative_hazard,
        risk_center=risk_center,
        fit_warnings=tuple(str(item.message) for item in caught),
    )


def baseline_hazard_at(
    event_times: NDArray[np.float64],
    cumulative_hazard: NDArray[np.float64],
    horizon: float,
) -> float:
    index = int(np.searchsorted(event_times, horizon, side="right")) - 1
    return 0.0 if index < 0 else float(cumulative_hazard[index])


def predict_survival_at_horizon(
    fitted: FittedCoxBaseline,
    transformed_predictors: NDArray[np.float64],
    horizon: float,
) -> NDArray[np.float64]:
    linear_predictor = transformed_predictors @ fitted.coefficients
    relative_hazard = np.exp(linear_predictor - fitted.risk_center)
    baseline_hazard = baseline_hazard_at(
        fitted.baseline_event_times,
        fitted.baseline_cumulative_hazard,
        horizon,
    )
    survival = np.exp(-baseline_hazard * relative_hazard)
    if not np.isfinite(survival).all():
        raise ValueError("Predicted survival contains non-finite values.")
    return survival


def preprocessing_summary(preprocessor: ColumnTransformer) -> dict[str, Any]:
    continuous = preprocessor.named_transformers_["continuous"]
    binary = preprocessor.named_transformers_["binary"]
    continuous_imputer = continuous.named_steps["imputer"]
    scaler = continuous.named_steps["scaler"]
    binary_imputer = binary.named_steps["imputer"]
    return {
        "fit_split": "train",
        "continuous_strategy": "median imputation followed by standard scaling",
        "binary_strategy": "most-frequent imputation without scaling",
        "missing_indicators_added": False,
        "continuous": {
            feature: {
                "imputation_median": float(continuous_imputer.statistics_[index]),
                "post_imputation_mean": float(scaler.mean_[index]),
                "post_imputation_scale": float(scaler.scale_[index]),
            }
            for index, feature in enumerate(CONTINUOUS_FEATURES)
        },
        "binary": {
            feature: {"imputation_mode": float(binary_imputer.statistics_[index])}
            for index, feature in enumerate(BINARY_FEATURES)
        },
    }


def coefficient_summary(fitted: FittedCoxBaseline) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for feature, coefficient, standard_error, p_value in zip(
        fitted.feature_names,
        fitted.coefficients,
        fitted.standard_errors,
        fitted.p_values,
        strict=True,
    ):
        lower = coefficient - 1.96 * standard_error
        upper = coefficient + 1.96 * standard_error
        rows.append(
            {
                "feature": feature,
                "coefficient": float(coefficient),
                "hazard_ratio": math.exp(float(coefficient)),
                "hazard_ratio_ci95_low": math.exp(float(lower)),
                "hazard_ratio_ci95_high": math.exp(float(upper)),
                "standard_error": float(standard_error),
                "p_value": float(p_value),
            }
        )
    return rows


def evaluate_splits(
    splits: CohortSplits,
    fitted: FittedCoxBaseline,
    *,
    horizon: float = EVALUATION_HORIZON_YEARS,
) -> dict[str, dict[str, float | int]]:
    train_duration = splits.train["duration_years"].to_numpy(dtype=float)
    train_event = splits.train["event"].to_numpy(dtype=np.int64)
    censoring_curve = kaplan_meier_curve(train_duration, 1 - train_event)
    censoring_survival_at_horizon = censoring_curve.survival_at(horizon)
    metrics: dict[str, dict[str, float | int]] = {}

    for split_name, frame in splits.as_dict().items():
        transformed = transform_predictors(fitted.preprocessor, frame)
        linear_predictor = transformed @ fitted.coefficients
        duration = frame["duration_years"].to_numpy(dtype=float)
        event = frame["event"].to_numpy(dtype=np.int64)
        predicted_survival = predict_survival_at_horizon(fitted, transformed, horizon)
        observed_risk = observed_event_risk_at_horizon(duration, event, horizon)
        predicted_risk = float(np.mean(1.0 - predicted_survival))

        metrics[split_name] = {
            "participants": int(len(frame)),
            "events_total": int(event.sum()),
            "event_percent": float(100.0 * event.mean()),
            "events_by_horizon": int(((event == 1) & (duration <= horizon)).sum()),
            "censored_by_horizon": int(((event == 0) & (duration <= horizon)).sum()),
            "observed_beyond_horizon": int((duration > horizon).sum()),
            "harrell_c_index": harrell_c_index(duration, event, linear_predictor),
            "ipcw_brier_at_horizon": ipcw_brier_score(
                train_duration,
                train_event,
                duration,
                event,
                predicted_survival,
                horizon,
            ),
            "mean_predicted_risk_at_horizon": predicted_risk,
            "km_observed_risk_at_horizon": observed_risk,
            "calibration_in_the_large_difference": predicted_risk - observed_risk,
            "training_censoring_survival_at_horizon": censoring_survival_at_horizon,
        }
    return metrics


def save_model_artifact(fitted: FittedCoxBaseline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": "cox_inference_artifact_v1",
            "model_name": MODEL_NAME,
            "predictor_features": PREDICTOR_FEATURES,
            "preprocessor": fitted.preprocessor,
            "feature_names": fitted.feature_names,
            "coefficients": fitted.coefficients,
            "baseline_event_times": fitted.baseline_event_times,
            "baseline_cumulative_hazard": fitted.baseline_cumulative_hazard,
            "risk_center": fitted.risk_center,
        },
        output_path,
    )
