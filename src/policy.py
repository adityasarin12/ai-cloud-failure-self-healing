from dataclasses import dataclass


@dataclass(frozen=True)
class ActionPolicy:
    name: str
    allowed: bool
    approval_required: bool
    max_blast_radius: int
    rollback_required: bool
    reason: str


POLICIES = {
    "Normal": (True, False, 0, False),
    "Scale Resources": (True, False, 2, True),
    "Migrate VM": (True, True, 3, True),
    "Restart Task": (True, True, 1, True),
}


def get_base_action(risk):
    """Map risk to exactly one action before safety and execution decisions."""
    risk = float(risk)
    if risk < 0.30:
        return "Normal"
    if risk < 0.50:
        return "Scale Resources"
    if risk < 0.65:
        return "Migrate VM"
    return "Restart Task"


def risk_actions(risk):
    """Return the single action permitted by the project's risk policy."""
    return {get_base_action(risk)}


def check_policy(action: str, risk: float, confidence_level: str, max_blast_radius: int = 1) -> ActionPolicy:
    allowed, approval_required, blast_radius, rollback_required = POLICIES.get(
        action, (False, True, 0, True)
    )
    reasons = []
    if action not in risk_actions(risk):
        allowed = False
        reasons.append("action is outside the risk-based action band")
    if blast_radius > max_blast_radius:
        allowed = False
        reasons.append("blast radius exceeds configured limit")
    if not reasons:
        reasons.append("action satisfies policy constraints")
    return ActionPolicy(action, allowed, approval_required, blast_radius, rollback_required, "; ".join(reasons))