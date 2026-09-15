from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Outcome:
    status: str
    recovery_time: float
    latency_change: float
    error_rate_change: float
    risk_change: float


def validate_outcome(before, after, recovery_time, success_threshold=0.0):
    risk_change = float(after["risk"] - before["risk"])
    latency_change = float(after["latency"] - before["latency"])
    error_rate_change = float(after["error_rate"] - before["error_rate"])
    improved = risk_change <= success_threshold and error_rate_change <= 0
    if improved and latency_change <= 0:
        status = "SUCCESS"
    elif improved:
        status = "PARTIAL_SUCCESS"
    else:
        status = "FAILURE"
    return Outcome(status, float(recovery_time), latency_change, error_rate_change, risk_change)


def outcome_record(incident, decision, outcome):
    return {"incident": incident, "action": decision["recommended_action"], **asdict(outcome)}