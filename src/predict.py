import pandas as pd
from src.confidence import assess_confidence
from src.decision_engine import default_action_estimates, make_decision
from src.risk_score import _normalize_anomaly_score, calculate_risk, self_heal


def generate_results(model, iso, X_test):

    failure_prob = model.predict_proba(X_test)[:, 1]
    raw_anomaly_score = iso.decision_function(X_test)
    calibrated_anomaly = _normalize_anomaly_score(raw_anomaly_score)

    risk = calculate_risk(failure_prob, raw_anomaly_score)
    confidence = [
        assess_confidence(probability, anomaly)
        for probability, anomaly in zip(failure_prob, raw_anomaly_score)
    ]

    results = pd.DataFrame({
        "failure_probability": failure_prob,
        "raw_anomaly_score": raw_anomaly_score,
        "anomaly_score": calibrated_anomaly,
        "risk_score": risk,
    })

    results["action"] = [
        make_decision(
            incident=f"batch_{index}",
            risk=risk_value,
            confidence=item,
            estimates=default_action_estimates(risk_value),
        )["recommended_action"]
        for index, (risk_value, item) in enumerate(zip(risk, confidence))
    ]
    results["confidence_score"] = [item.score for item in confidence]
    results["confidence_level"] = [item.level for item in confidence]

    return results