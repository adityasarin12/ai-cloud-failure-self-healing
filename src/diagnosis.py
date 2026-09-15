import numpy as np
from src.risk_score import _normalize_anomaly_score


def build_signature(features, feature_names=None):
    values = np.asarray(features, dtype=float).reshape(-1)
    return values.tolist()


def diagnose(model, features, feature_names=None, top_k=5):
    """Return model-driver evidence, not a causal claim."""
    names = list(getattr(model, "feature_names_in_", []) if feature_names is None else feature_names)
    values = np.asarray(features, dtype=float).reshape(-1)
    importances = np.asarray(getattr(model, "feature_importances_", np.zeros(len(values))))
    count = min(len(names), len(values), len(importances))
    ranked = sorted(
        zip(names[:count], values[:count], importances[:count]),
        key=lambda item: item[2],
        reverse=True,
    )
    return {
        "type": "model_driver_evidence",
        "drivers": [
            {"feature": name, "value": float(value), "importance": float(importance)}
            for name, value, importance in ranked[:top_k]
        ],
        "warning": "Drivers indicate predictive association, not proven root cause.",
    }


def explain_prediction(
    model,
    features,
    anomaly_model=None,
    feature_names=None,
    failure_probability=None,
    anomaly_signal=None,
    raw_anomaly_score=None,
    risk_score=None,
    recommended_action=None,
    top_k=5,
):
    """Explain model sensitivity using feature replacement, not causal claims.

    Each feature is replaced by zero in an otherwise identical row.  The
    resulting change is an input-specific model association for this row.
    """
    import pandas as pd

    values = np.asarray(features, dtype=float).reshape(-1)
    names = list(getattr(model, "feature_names_in_", []) if feature_names is None else feature_names)
    if len(names) != len(values):
        names = names[:len(values)]
    row = pd.DataFrame([values], columns=names)
    current_failure = float(model.predict_proba(row)[:, 1][0])
    current_anomaly = float(anomaly_signal if anomaly_signal is not None else 0.0)
    factors = []
    for index, name in enumerate(names):
        replaced = values.copy()
        replaced[index] = 0.0
        replaced_row = pd.DataFrame([replaced], columns=names)
        replacement_failure = float(model.predict_proba(replaced_row)[:, 1][0])
        failure_impact = current_failure - replacement_failure
        anomaly_impact = 0.0
        if anomaly_model is not None:
            replacement_raw = float(anomaly_model.decision_function(replaced_row)[0])
            anomaly_impact = current_anomaly - float(_normalize_anomaly_score(replacement_raw))
        impact = 0.7 * failure_impact + 0.3 * anomaly_impact
        factors.append({
            "feature": name,
            "value": float(values[index]),
            "impact": round(float(impact), 6),
            "direction": "increases_risk" if impact >= 0 else "decreases_risk",
            "failure_impact": round(float(failure_impact), 6),
            "anomaly_impact": round(float(anomaly_impact), 6),
        })
    factors.sort(key=lambda item: abs(item["impact"]), reverse=True)
    top_factors = factors[:top_k]
    if top_factors:
        strongest = top_factors[0]
        primary_reason = (
            f"{strongest['feature']} shows the strongest model-associated effect "
            f"and {strongest['direction'].replace('_', ' ')}."
        )
    else:
        primary_reason = "No feature-level model association was available."
    return {
        "primary_reason": primary_reason,
        "contributing_factors": top_factors,
        "failure_probability": float(current_failure if failure_probability is None else failure_probability),
        "anomaly_signal": float(current_anomaly),
        "risk_score": None if risk_score is None else float(risk_score),
        "recommended_action": recommended_action,
        "anomaly_evidence": {
            "raw_anomaly_score": None if raw_anomaly_score is None else float(raw_anomaly_score),
            "calibrated_anomaly_signal": float(current_anomaly),
            "features_associated_with_anomaly": [item["feature"] for item in top_factors if item["anomaly_impact"] != 0.0],
        },
        "disclaimer": "Model-based explanation — not causal proof.",
    }