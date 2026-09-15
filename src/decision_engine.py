from dataclasses import dataclass, asdict

from src.policy import check_policy, get_base_action


@dataclass(frozen=True)
class ActionEstimate:
    action: str
    post_failure_risk: float
    recovery_time: float
    latency_impact: float
    resource_cost: float
    historical_success: float
    reversibility: float


def default_action_estimates(risk):
    """Controlled baseline estimates until real post-action telemetry is available."""
    severity = max(0.0, min(1.0, float(risk)))
    return [
        ActionEstimate("Normal", severity, 1.0, 0.0, 0.0, 0.50, 1.0),
        ActionEstimate("Scale Resources", severity * 0.30, 0.5, 0.05, 0.6, 0.80, 0.80),
        ActionEstimate("Migrate VM", severity * 0.25, 0.9, 0.20, 0.8, 0.65, 0.60),
        ActionEstimate("Restart Task", severity * 0.45, 0.4, 0.10, 0.2, 0.75, 0.90),
    ]


def _confidence_details(confidence):
    if hasattr(confidence, "score"):
        return float(confidence.score), confidence.level
    score = max(0.0, min(1.0, float(confidence)))
    if score >= 0.80:
        return score, "HIGH"
    if score >= 0.55:
        return score, "MEDIUM"
    return score, "LOW"


def _runtime_reason(
    failure_probability,
    calibrated_anomaly,
    risk,
    risk_band,
    action,
    confidence_score,
    confidence_level,
    execution_mode,
    policy_status,
):
    """Explain a structured decision from the values produced for this request."""
    return (
        f"Failure probability {float(failure_probability):.3f} and calibrated anomaly "
        f"signal {float(calibrated_anomaly):.3f} produced risk {float(risk):.3f}, "
        f"which falls in the {risk_band} risk band. Selected action: {action}; "
        f"the policy maps this risk band to that action. Confidence is "
        f"{confidence_level} ({float(confidence_score):.3f}), so execution mode is "
        f"{execution_mode}. Policy result: {policy_status}."
    )


def rank_actions(risk, confidence, estimates, max_blast_radius=3):
    """Rank actions by normalized expected harm, reliability, and reversibility."""
    _, confidence_level = _confidence_details(confidence)
    base_action = get_base_action(risk)
    ranked = []
    for estimate in estimates:
        if estimate.action != base_action:
            continue
        policy = check_policy(estimate.action, risk, confidence_level, max_blast_radius)
        if not policy.allowed:
            continue
        score = (
            0.35 * estimate.post_failure_risk
            + 0.20 * estimate.recovery_time
            + 0.15 * estimate.latency_impact
            + 0.10 * estimate.resource_cost
            + 0.15 * (1.0 - estimate.historical_success)
            + 0.05 * (1.0 - estimate.reversibility)
        )
        ranked.append({
            **asdict(estimate),
            "score": round(float(score), 6),
            "policy_status": "approval_required" if policy.approval_required else "approved",
            "rollback_required": policy.rollback_required,
            "policy_reason": policy.reason,
        })
    return sorted(ranked, key=lambda item: item["score"])


def make_decision(
    incident,
    risk,
    confidence,
    estimates,
    similar_incidents=None,
    failure_probability=None,
    calibrated_anomaly=None,
):
    confidence_score, confidence_level = _confidence_details(confidence)
    base_action = get_base_action(risk)
    ranked = rank_actions(risk, confidence, estimates)
    if not ranked:
        action = base_action
        status = "blocked"
        reason = f"Risk policy selected {base_action}, but its policy checks blocked execution"
        rollback_required = False
    else:
        selected = ranked[0]
        action = selected["action"]
        status = selected["policy_status"]
        reason = f"Lowest expected harm score among policy-eligible actions: {selected['score']}"
        rollback_required = selected["rollback_required"]
    if confidence_level == "LOW":
        execution_mode = "ALERT_ONLY"
    elif confidence_level == "MEDIUM":
        execution_mode = "SUPERVISED"
    elif status == "approval_required":
        execution_mode = "SUPERVISED"
    else:
        execution_mode = "AUTONOMOUS"
    if failure_probability is not None and calibrated_anomaly is not None:
        reason = _runtime_reason(
            failure_probability,
            calibrated_anomaly,
            risk,
            base_action,
            action,
            confidence_score,
            confidence_level,
            execution_mode,
            status,
        )
    return {
        "incident": incident,
        "risk": round(float(risk), 6),
        "risk_band": base_action,
        "confidence": round(confidence_score, 6),
        "confidence_level": confidence_level,
        "recommended_action": action,
        "execution_mode": execution_mode,
        "alternatives": ranked[1:],
        "similar_incidents": similar_incidents or [],
        "reason": reason,
        "expected_recovery": ranked[0]["recovery_time"] if ranked else None,
        "policy_status": status,
        "rollback_required": rollback_required,
    }
