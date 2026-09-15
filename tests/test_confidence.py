from src.confidence import assess_confidence
from src.decision_engine import default_action_estimates, make_decision


def test_confidence_levels_map_to_execution_modes():
    assert make_decision("high", 0.40, 0.95, default_action_estimates(0.40))["execution_mode"] == "AUTONOMOUS"
    assert make_decision("medium", 0.40, 0.70, default_action_estimates(0.40))["execution_mode"] == "SUPERVISED"
    assert make_decision("low", 0.40, 0.40, default_action_estimates(0.40))["execution_mode"] == "ALERT_ONLY"


def test_model_confidence_object_is_supported():
    confidence = assess_confidence(0.9, -0.13)
    decision = make_decision("object", 0.70, confidence, default_action_estimates(0.70))
    assert decision["recommended_action"] == "Restart Task"
    assert decision["confidence_level"] == confidence.level
