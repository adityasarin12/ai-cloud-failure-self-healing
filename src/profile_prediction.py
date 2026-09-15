"""Training-faithful inputs for the Single Prediction UI.

The saved estimators were trained on preprocessed Borg task rows.  This module
keeps a selected reference row intact and only permits explicit edits to fields
that represent direct task telemetry.  It deliberately does not synthesize
related CPU or memory measurements and never fills model fields with medians.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.preprocessing import preprocess_data


# These are task configuration or observed telemetry values.  Identifiers and
# one-hot scheduling metadata are intentionally not user-editable.
EDITABLE_TELEMETRY_FEATURES = (
    "priority",
    "assigned_memory",
    "page_cache_memory",
    "cycles_per_instruction",
    "memory_accesses_per_instruction",
    "sample_rate",
    "req_cpu",
    "req_memory",
    "avg_cpu",
    "avg_memory",
    "max_cpu",
    "max_memory",
    "sample_cpu",
    "cpu_mean",
    "tail_cpu_mean",
)

IMPORTANT_TELEMETRY_FEATURES = EDITABLE_TELEMETRY_FEATURES


def load_reference_profiles(data_path: str, model_features: Sequence[str], nrows: int = 5_000) -> pd.DataFrame:
    """Return real preprocessed task rows in the saved-model feature order."""
    prepared = preprocess_data(data_path, nrows=nrows)
    missing = [feature for feature in model_features if feature not in prepared.columns]
    if missing:
        raise ValueError(f"Preprocessed reference data is missing model features: {missing}")
    return prepared.loc[:, list(model_features)].copy()


def build_profile_prediction_features(
    profile: pd.Series | Mapping[str, object],
    model_features: Sequence[str],
    edits: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Create one exact model row from a real profile and optional direct edits."""
    values = pd.Series(profile).copy()
    missing = [feature for feature in model_features if feature not in values.index]
    if missing:
        raise ValueError(f"Selected profile is missing model features: {missing}")

    if edits:
        unsupported = set(edits).difference(EDITABLE_TELEMETRY_FEATURES)
        if unsupported:
            raise ValueError(f"Only direct telemetry fields may be edited: {sorted(unsupported)}")
        for feature, value in edits.items():
            if feature not in model_features:
                raise ValueError(f"Editable feature is absent from this model: {feature}")
            values[feature] = value

    # This explicit selection is the contract with the persisted estimators.
    return pd.DataFrame([values.loc[list(model_features)].to_dict()], columns=list(model_features))


def manual_telemetry_specs(
    reference_profiles: pd.DataFrame,
    selected_profile: pd.Series | Mapping[str, object] | None = None,
) -> dict[str, dict[str, float]]:
    """Build editable slider bounds from reference quantiles and profile defaults."""
    specs = {}
    profile_values = pd.Series(selected_profile) if selected_profile is not None else None
    for feature in EDITABLE_TELEMETRY_FEATURES:
        if feature not in reference_profiles.columns:
            continue
        values = pd.to_numeric(reference_profiles[feature], errors="coerce").dropna()
        if values.empty:
            continue
        low, high = values.quantile([0.01, 0.99])
        if np.isclose(low, high):
            low, high = values.min(), values.max()
        if np.isclose(low, high):
            continue
        default = values.median() if profile_values is None else profile_values[feature]
        default = float(np.clip(float(default), low, high))
        step = max(float(high - low) / 100.0, np.finfo(float).eps)
        decimals = max(2, int(np.ceil(-np.log10(step))) + 1)
        specs[feature] = {
            "min": float(low),
            "max": float(high),
            "default": default,
            "step": step,
            "format": f"%.{decimals}f",
            "reference_min": float(values.min()),
            "reference_p01": float(values.quantile(0.01)),
            "reference_median": float(values.median()),
            "reference_p99": float(values.quantile(0.99)),
            "reference_max": float(values.max()),
        }
    return specs


def profile_distribution_diagnostic(profile: pd.Series | Mapping[str, object], reference_profiles: pd.DataFrame) -> pd.DataFrame:
    """Report reference distribution statistics and OOD status for one input row."""
    values = pd.Series(profile)
    rows = []
    for feature in reference_profiles.columns:
        reference = pd.to_numeric(reference_profiles[feature], errors="coerce").dropna().astype(float)
        value = float(pd.to_numeric(values[feature], errors="coerce"))
        rows.append({
            "feature": feature,
            "reference_min": reference.min(),
            "reference_p01": reference.quantile(0.01),
            "reference_max": reference.max(),
            "reference_median": reference.median(),
            "reference_p99": reference.quantile(0.99),
            "profile_value": value,
            "within_reference_distribution": bool(reference.min() <= value <= reference.max()),
            "distribution_status": (
                "IN RANGE" if reference.quantile(0.01) <= value <= reference.quantile(0.99)
                else "LOW OOD" if value < reference.quantile(0.01) else "HIGH OOD"
            ),
        })
    return pd.DataFrame(rows)


def input_diagnostics(
    input_row: pd.Series | Mapping[str, object], reference_profiles: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Return central-98%-range OOD count and highest-distance feature names."""
    diagnostic = profile_distribution_diagnostic(input_row, reference_profiles)
    numeric = diagnostic.copy()
    numeric["distance"] = np.where(
        numeric["profile_value"] < numeric["reference_p01"],
        numeric["reference_p01"] - numeric["profile_value"],
        np.where(
            numeric["profile_value"] > numeric["reference_p99"],
            numeric["profile_value"] - numeric["reference_p99"],
            0.0,
        ),
    )
    top = numeric.sort_values("distance", ascending=False)
    outside = top.loc[top["distance"] > 0, "feature"].tolist()
    return diagnostic, outside


def telemetry_distribution_audit(
    reference_profiles: pd.DataFrame,
    selected_profile: pd.Series | Mapping[str, object],
) -> pd.DataFrame:
    """Compare each editable field's training distribution with its UI range."""
    specs = manual_telemetry_specs(reference_profiles, selected_profile)
    rows = []
    for feature, spec in specs.items():
        profile_value = float(pd.to_numeric(pd.Series(selected_profile)[feature], errors="coerce"))
        rows.append({
            "feature": feature,
            "training_min": spec["reference_min"],
            "training_p01": spec["reference_p01"],
            "training_median": spec["reference_median"],
            "training_p99": spec["reference_p99"],
            "training_max": spec["reference_max"],
            "ui_min": spec["min"],
            "ui_max": spec["max"],
            "ui_default": spec["default"],
            "selected_profile_value": profile_value,
            "status": "IN_DISTRIBUTION" if spec["min"] <= profile_value <= spec["max"] else "OUT_OF_DISTRIBUTION",
        })
    return pd.DataFrame(rows)
