import json

import pytest

from src.decision_engine import default_action_estimates, make_decision, rank_actions
from src.policy import POLICIES, check_policy


@pytest.mark.parametrize(
    "risk,expected_action",
    [
        (0.00, "Normal"), (0.10, "Normal"), (0.20, "Normal"), (0.29, "Normal"),
        (0.30, "Scale Resources"), (0.35, "Scale Resources"), (0.391, "Scale Resources"),
        (0.40, "Scale Resources"), (0.49, "Scale Resources"),
        (0.50, "Migrate VM"), (0.55, "Migrate VM"), (0.60, "Migrate VM"), (0.64, "Migrate VM"),
        (0.65, "Restart Task"), (0.70, "Restart Task"), (0.80, "Restart Task"),
        (0.90, "Restart Task"), (1.00, "Restart Task"),
        (0.299999, "Normal"), (0.300000, "Scale Resources"),
        (0.499999, "Scale Resources"), (0.500000, "Migrate VM"),
        (0.649999, "Migrate VM"), (0.650000, "Restart Task"),
    ],
)
def test_actual_decision_engine_respects_risk_bands(risk, expected_action):
    decision = make_decision("regression", risk, 0.95, default_action_estimates(risk))
    assert decision["recommended_action"] == expected_action


@pytest.mark.parametrize(
    "risk,expected_action",
    [(0.469, "Scale Resources"), (0.40, "Scale Resources"),
     (0.55, "Migrate VM"), (0.70, "Restart Task")],
)
def test_confidence_cannot_override_risk_action(risk, expected_action):
    for confidence in (0.40, 0.70, 0.95):
        decision = make_decision("confidence", risk, confidence, default_action_estimates(risk))
        assert decision["recommended_action"] == expected_action


def test_allowed_action_set_is_exactly_four_actions():
    assert set(POLICIES) == {"Normal", "Scale Resources", "Migrate VM", "Restart Task"}


def test_alternatives_are_valid_actions():
    decision = make_decision("alternatives", 0.40, 0.95, default_action_estimates(0.40))
    allowed = set(POLICIES)
    assert decision["recommended_action"] in allowed
    assert all(item["action"] in allowed for item in decision["alternatives"])


def test_decision_is_json_serializable_and_has_required_fields():
    decision = make_decision("json", 0.55, 0.95, default_action_estimates(0.55))
    assert {"risk", "risk_band", "confidence", "confidence_level", "recommended_action", "execution_mode", "reason"} <= set(decision)
    json.dumps(decision)


def test_runtime_reason_uses_values_and_changes_with_risk_action():
    normal = make_decision(
        "normal", 0.2, 0.8, default_action_estimates(0.2),
        failure_probability=0.1, calibrated_anomaly=0.433333,
    )
    restart = make_decision(
        "restart", 0.8, 0.9, default_action_estimates(0.8),
        failure_probability=0.9, calibrated_anomaly=0.566667,
    )
    assert normal["reason"] and restart["reason"]
    assert "Failure probability 0.100" in normal["reason"]
    assert "Normal risk band" in normal["reason"]
    assert "Restart Task" in restart["reason"]
    assert normal["reason"] != restart["reason"]


def test_blocked_action_is_explicit_and_not_replaced():
    policy = check_policy("Migrate VM", 0.55, "HIGH", max_blast_radius=1)
    assert not policy.allowed
    assert "blast radius" in policy.reason
    assert rank_actions(0.55, 0.95, default_action_estimates(0.55), max_blast_radius=1) == []
    decision = make_decision("blocked", 0.55, 0.95, [])
    assert decision["recommended_action"] == "Migrate VM"
    assert decision["policy_status"] == "blocked"
