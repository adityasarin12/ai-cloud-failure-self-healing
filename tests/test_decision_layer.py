import unittest

from src.confidence import assess_confidence
from src.decision_engine import ActionEstimate, default_action_estimates, make_decision, rank_actions
from src.healing import DryRunHealingAdapter, request_from_decision
from src.outcome import validate_outcome
from src.risk_score import calculate_risk


class DecisionLayerTests(unittest.TestCase):
    def test_risk_is_bounded_for_isolation_forest_scores(self):
        self.assertGreaterEqual(calculate_risk(0.9, -0.13), 0.0)
        self.assertLessEqual(calculate_risk(0.9, -0.13), 1.0)

    def test_risk_band_controls_action_not_confidence(self):
        estimates = default_action_estimates(0.4)
        low = make_decision("low", 0.4, 0.40, estimates)
        high = make_decision("high", 0.4, 0.90, estimates)
        self.assertEqual(low["recommended_action"], "Scale Resources")
        self.assertEqual(low["recommended_action"], high["recommended_action"])
        self.assertEqual(low["execution_mode"], "ALERT_ONLY")
        self.assertEqual(high["execution_mode"], "AUTONOMOUS")

    def test_intended_risk_bands(self):
        expected = [
            (0.00, "Normal"), (0.20, "Normal"), (0.29, "Normal"),
            (0.30, "Scale Resources"), (0.40, "Scale Resources"), (0.49, "Scale Resources"),
            (0.50, "Migrate VM"), (0.55, "Migrate VM"), (0.64, "Migrate VM"),
            (0.65, "Restart Task"), (0.70, "Restart Task"), (0.90, "Restart Task"),
            (1.00, "Restart Task"),
            (0.299999, "Normal"), (0.300000, "Scale Resources"),
            (0.499999, "Scale Resources"), (0.500000, "Migrate VM"),
            (0.649999, "Migrate VM"), (0.650000, "Restart Task"),
        ]
        for risk, action in expected:
            decision = make_decision("band", risk, 0.90, default_action_estimates(risk))
            self.assertEqual(decision["recommended_action"], action, risk)

    def test_high_confidence_does_not_override_risk_action(self):
        for risk, action in ((0.40, "Scale Resources"), (0.55, "Migrate VM"), (0.70, "Restart Task")):
            decision = make_decision("confidence", risk, 0.95, default_action_estimates(risk))
            self.assertEqual(decision["recommended_action"], action, risk)

    def test_blocked_policy_keeps_base_action(self):
        decision = make_decision("blocked", 0.55, 0.95, [])
        self.assertEqual(decision["recommended_action"], "Migrate VM")
        self.assertEqual(decision["policy_status"], "blocked")

    def test_policy_ranked_decision_becomes_dry_run_request(self):
        confidence = assess_confidence(0.9, -0.13)
        estimates = [ActionEstimate("Restart Task", 0.2, 0.2, 0.1, 0.2, 0.9, 0.9)]
        decision = make_decision("incident-1", 0.8, confidence, estimates)
        request = request_from_decision(decision)
        result = DryRunHealingAdapter().execute(request)
        self.assertEqual(result["status"], "DRY_RUN")
        self.assertEqual(result["action"], decision["recommended_action"])
        self.assertTrue(request.rollback_required)

    def test_outcome_classification(self):
        outcome = validate_outcome(
            {"risk": 0.8, "latency": 20, "error_rate": 0.2},
            {"risk": 0.2, "latency": 18, "error_rate": 0.1},
            recovery_time=4,
        )
        self.assertEqual(outcome.status, "SUCCESS")


if __name__ == "__main__":
    unittest.main()