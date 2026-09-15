from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceResult:
    score: float
    level: str
    autonomy: str
    reason: str


def assess_confidence(
    failure_probability: float,
    anomaly_score: float,
    high_threshold: float = 0.80,
    medium_threshold: float = 0.55,
) -> ConfidenceResult:
    """Use model agreement as a lightweight, real-time confidence proxy.

    Failure probability is already bounded. Isolation Forest scores are not,
    so agreement is measured after mapping anomaly evidence to [0, 1].
    """
    from src.risk_score import _normalize_anomaly_score

    anomaly_signal = float(_normalize_anomaly_score(anomaly_score))
    failure_probability = max(0.0, min(1.0, float(failure_probability)))
    agreement = 1.0 - abs(failure_probability - anomaly_signal)
    score = max(0.0, min(1.0, agreement))

    if score >= high_threshold:
        return ConfidenceResult(score, "HIGH", "autonomous", "Models agree strongly")
    if score >= medium_threshold:
        return ConfidenceResult(score, "MEDIUM", "supervised", "Models show partial agreement")
    return ConfidenceResult(score, "LOW", "alert_only", "Models disagree")