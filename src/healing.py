from dataclasses import dataclass


@dataclass(frozen=True)
class HealingRequest:
    incident: str
    action: str
    approval_required: bool
    rollback_required: bool
    parameters: dict


class CloudHealingAdapter:
    """Contract for the teammate's AWS/Kubernetes executor."""

    def execute(self, request: HealingRequest):
        raise NotImplementedError


class DryRunHealingAdapter(CloudHealingAdapter):
    """Safe local adapter that records intent without touching infrastructure."""

    def execute(self, request: HealingRequest):
        return {"status": "DRY_RUN", "action": request.action, "incident": request.incident}


def request_from_decision(decision, parameters=None):
    selected = next(
        (item for item in [*decision.get("alternatives", []), {
            "action": decision["recommended_action"],
            "policy_status": decision["policy_status"],
            "rollback_required": decision.get("rollback_required", False),
        }] if item["action"] == decision["recommended_action"]),
        None,
    )
    return HealingRequest(
        incident=decision["incident"],
        action=decision["recommended_action"],
        approval_required=decision["policy_status"] == "approval_required",
        rollback_required=bool(selected and selected.get("rollback_required", False)),
        parameters=parameters or {},
    )