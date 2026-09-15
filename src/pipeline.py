from src.confidence import assess_confidence
from src.decision_engine import default_action_estimates, make_decision
from src.diagnosis import build_signature, diagnose
from src.diagnosis import explain_prediction
from src.risk_score import _normalize_anomaly_score, calculate_risk
import time


def run_decision_pipeline(model, anomaly_model, features, incident_id, memory=None):
    """Run prediction through diagnosis, incident memory, policy, and ranking."""
    total_started = time.perf_counter()
    row = features.iloc[[0]] if hasattr(features, "iloc") else features
    preprocessing_started = time.perf_counter()
    preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000
    failure_started = time.perf_counter()
    failure_probability = float(model.predict_proba(row)[:, 1][0])
    failure_model_ms = (time.perf_counter() - failure_started) * 1000
    anomaly_started = time.perf_counter()
    raw_anomaly_score = float(anomaly_model.decision_function(row)[0])
    anomaly_model_ms = (time.perf_counter() - anomaly_started) * 1000
    calibration_started = time.perf_counter()
    calibrated_anomaly = float(_normalize_anomaly_score(raw_anomaly_score))
    calibration_ms = (time.perf_counter() - calibration_started) * 1000
    risk_started = time.perf_counter()
    risk = calculate_risk(failure_probability, raw_anomaly_score)
    confidence = assess_confidence(failure_probability, raw_anomaly_score)
    signature = build_signature(row.to_numpy())
    similar = memory.search(signature) if memory else []
    decision = make_decision(
        incident=incident_id,
        risk=risk,
        confidence=confidence,
        estimates=default_action_estimates(risk),
        similar_incidents=similar,
        failure_probability=failure_probability,
        calibrated_anomaly=calibrated_anomaly,
    )
    risk_decision_ms = (time.perf_counter() - risk_started) * 1000
    explanation_started = time.perf_counter()
    decision["diagnosis"] = diagnose(model, row.to_numpy()[0], feature_names=row.columns)
    decision["explanation"] = explain_prediction(
        model,
        row.to_numpy()[0],
        anomaly_model=anomaly_model,
        feature_names=row.columns,
        failure_probability=failure_probability,
        anomaly_signal=calibrated_anomaly,
        raw_anomaly_score=raw_anomaly_score,
        risk_score=risk,
        recommended_action=decision["recommended_action"],
    )
    explanation_ms = (time.perf_counter() - explanation_started) * 1000
    decision["failure_probability"] = failure_probability
    decision["raw_anomaly_score"] = raw_anomaly_score
    decision["calibrated_anomaly"] = calibrated_anomaly
    decision["anomaly_score"] = calibrated_anomaly
    decision["risk"] = risk
    decision["latency"] = {
        "preprocessing_ms": preprocessing_ms,
        "failure_model_ms": failure_model_ms,
        "anomaly_model_ms": anomaly_model_ms,
        "calibration_ms": calibration_ms,
        "risk_decision_ms": risk_decision_ms,
        "explanation_ms": explanation_ms,
        "forecast_ms": 0.0,
        "total_inference_ms": (time.perf_counter() - total_started) * 1000,
    }
    return decision
