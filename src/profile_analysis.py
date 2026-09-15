"""Empirical action audit for real Single Prediction reference profiles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from src.confidence import assess_confidence
from src.decision_engine import default_action_estimates, make_decision
from src.risk_score import _normalize_anomaly_score, calculate_risk
from src.pipeline import run_decision_pipeline
from src.policy import get_base_action
from src.profile_prediction import (
    EDITABLE_TELEMETRY_FEATURES,
    build_profile_prediction_features,
    manual_telemetry_specs,
)


ANALYSIS_COLUMNS = (
    "profile_index",
    "failure_probability",
    "raw_isolation_forest_score",
    "calibrated_anomaly_score",
    "risk_score",
    "risk_band",
    "final_recommended_action",
    "confidence",
    "confidence_level",
)


def analyze_profile_actions(model, anomaly_model, profiles: pd.DataFrame) -> pd.DataFrame:
    """Run each real profile through the exact Single Prediction inference path.

    Each iteration uses the same full-row builder and ``run_decision_pipeline``
    called by the Streamlit UI.  No profile values are changed or generated.
    """
    rows = []
    for profile_index, profile in profiles.iterrows():
        features = build_profile_prediction_features(profile, model.feature_names_in_)
        decision = run_decision_pipeline(
            model, anomaly_model, features, incident_id=f"reference-profile-{profile_index}"
        )
        risk = float(decision["risk"])
        rows.append({
            "profile_index": int(profile_index),
            "failure_probability": float(decision["failure_probability"]),
            "raw_isolation_forest_score": float(decision["raw_anomaly_score"]),
            "calibrated_anomaly_score": float(decision["calibrated_anomaly"]),
            "risk_score": risk,
            "risk_band": get_base_action(risk),
            "final_recommended_action": decision["recommended_action"],
            "confidence": float(decision["confidence"]),
            "confidence_level": decision["confidence_level"],
        })
    return pd.DataFrame(rows, columns=ANALYSIS_COLUMNS)


def write_profile_action_analysis(results: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist auditable evidence without changing the model artifacts."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    return path


def action_distribution(results: pd.DataFrame) -> pd.DataFrame:
    """Return all policy actions with count and percentage, including zeroes."""
    actions = ("Normal", "Scale Resources", "Migrate VM", "Restart Task")
    counts = results["final_recommended_action"].value_counts().reindex(actions, fill_value=0)
    return pd.DataFrame({
        "Action": actions,
        "Count": counts.to_numpy(),
        "Percentage": (counts.to_numpy() * 100 / len(results)).round(2),
    })


def risk_summary(results: pd.DataFrame) -> pd.Series:
    """Required deterministic risk percentiles for audit reporting."""
    return results["risk_score"].quantile([0, 0.01, 0.25, 0.50, 0.75, 0.99, 1.0])


def first_profile_per_action(results: pd.DataFrame) -> pd.DataFrame:
    """First naturally occurring real profile for each observed final action."""
    return (
        results.sort_values("profile_index")
        .groupby("final_recommended_action", as_index=False)
        .first()
    )


def analyze_manual_telemetry(
    model, anomaly_model, profiles: pd.DataFrame, sample_count: int = 500, seed: int = 42
) -> pd.DataFrame:
    """Audit independent Manual Telemetry edits across reference quantile ranges."""
    selected = profiles.iloc[0]
    specs = manual_telemetry_specs(profiles, selected)
    names = [feature for feature in EDITABLE_TELEMETRY_FEATURES if feature in specs]
    cases = [{feature: specs[feature]["default"] for feature in names}]
    cases.append({feature: specs[feature]["min"] for feature in names})
    cases.append({feature: specs[feature]["max"] for feature in names})
    rng = np.random.default_rng(seed)
    for values in rng.uniform(
        [specs[feature]["min"] for feature in names],
        [specs[feature]["max"] for feature in names],
        size=(sample_count, len(names)),
    ):
        cases.append(dict(zip(names, values)))

    inputs = pd.concat(
        [build_profile_prediction_features(selected, model.feature_names_in_, edits=edits) for edits in cases],
        ignore_index=True,
    )
    failure_probabilities = model.predict_proba(inputs)[:, 1]
    raw_anomaly_scores = anomaly_model.decision_function(inputs)
    anomaly_signals = _normalize_anomaly_score(raw_anomaly_scores)
    risks = calculate_risk(failure_probabilities, raw_anomaly_scores)
    rows = []
    for index, (failure_probability, raw_anomaly, anomaly_signal, risk) in enumerate(
        zip(failure_probabilities, raw_anomaly_scores, anomaly_signals, risks)
    ):
        confidence = assess_confidence(failure_probability, raw_anomaly)
        decision = make_decision(
            incident=f"manual-telemetry-{index}",
            risk=risk,
            confidence=confidence,
            estimates=default_action_estimates(risk),
        )
        rows.append({
            "case": index,
            **inputs.iloc[index].to_dict(),
            "failure_probability": float(failure_probability),
            "raw_isolation_forest_score": float(raw_anomaly),
            "calibrated_anomaly_signal": float(anomaly_signal),
            "risk_score": float(risk),
            "risk_band": get_base_action(risk),
            "confidence": float(confidence.score),
            "action": decision["recommended_action"],
        })
    return pd.DataFrame(rows)
