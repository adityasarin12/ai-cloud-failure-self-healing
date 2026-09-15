from pathlib import Path

import numpy as np
import pandas as pd

from src.confidence import assess_confidence
from src.decision_engine import default_action_estimates, make_decision
from src.policy import get_base_action
from src.risk_score import _normalize_anomaly_score, calculate_risk
from src.single_prediction import UI_FEATURE_RANGES, build_single_prediction_features


def representative_ui_inputs(random_count=2000, seed=42):
    minimum = {name: bounds[0] for name, bounds in UI_FEATURE_RANGES.items()}
    maximum = {name: bounds[1] for name, bounds in UI_FEATURE_RANGES.items()}
    low_cpu_memory = {**minimum, "priority": 500, "cpu_mean": 0.1, "assigned_memory": 0.1}
    high_cpu_memory = {**maximum, "priority": 0, "cpu_mean": 0.9, "assigned_memory": 0.9}
    low_cpi = {**maximum, "cycles_per_instruction": 0.0}
    high_cpi = {**minimum, "cycles_per_instruction": 2.0}
    low_accesses = {**maximum, "memory_accesses_per_instruction": 0.0}
    high_accesses = {**minimum, "memory_accesses_per_instruction": 1.0}
    cases = [
        ("minimum", minimum), ("maximum", maximum),
        ("low_cpu_memory", low_cpu_memory), ("high_cpu_memory", high_cpu_memory),
        ("low_cpi", low_cpi), ("high_cpi", high_cpi),
        ("low_memory_accesses", low_accesses), ("high_memory_accesses", high_accesses),
    ]
    rng = np.random.default_rng(seed)
    lower = np.array([bounds[0] for bounds in UI_FEATURE_RANGES.values()])
    upper = np.array([bounds[1] for bounds in UI_FEATURE_RANGES.values()])
    names = list(UI_FEATURE_RANGES)
    for index, values in enumerate(rng.uniform(lower, upper, size=(random_count, len(names)))):
        cases.append((f"random_{index:04d}", dict(zip(names, values))))
    return cases


def analyze_single_prediction_boundary(model, anomaly_model, feature_medians, random_count=2000, seed=42):
    cases = representative_ui_inputs(random_count, seed)
    values_list = [values for _, values in cases]
    features = pd.concat(
        [build_single_prediction_features(values, model.feature_names_in_, feature_medians)
         for values in values_list],
        ignore_index=True,
    )
    failure_probabilities = model.predict_proba(features)[:, 1]
    raw_anomaly_scores = anomaly_model.decision_function(features)
    calibrated_signals = _normalize_anomaly_score(raw_anomaly_scores)
    risks = calculate_risk(failure_probabilities, raw_anomaly_scores)
    rows = []
    for (case, values), failure_probability, raw_anomaly, calibrated_anomaly, risk in zip(
        cases, failure_probabilities, raw_anomaly_scores, calibrated_signals, risks
    ):
        confidence = assess_confidence(failure_probability, raw_anomaly)
        decision = make_decision(
            incident=f"boundary-{case}",
            risk=risk,
            confidence=confidence,
            estimates=default_action_estimates(risk),
        )
        rows.append({
            "case": case,
            **values,
            "failure_probability": float(failure_probability),
            "raw_isolation_forest_score": float(raw_anomaly),
            "calibrated_anomaly_signal": float(calibrated_anomaly),
            "risk_score": risk,
            "risk_band": get_base_action(risk),
            "base_action": get_base_action(risk),
            "final_action": decision["recommended_action"],
            "confidence": confidence.score,
        })
    return pd.DataFrame(rows)


def write_boundary_report(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return output_path