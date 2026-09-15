import os
import joblib
import numpy as np
import pytest

from src.risk_score import (
    AnomalyCalibrator,
    _normalize_anomaly_score,
    calculate_risk,
    get_anomaly_calibrator
)
from src.decision_engine import default_action_estimates, make_decision
from src.policy import get_base_action, POLICIES


def test_calibrator_artifact_exists_and_is_loaded():
    calibrator = get_anomaly_calibrator()
    assert calibrator is not None
    assert isinstance(calibrator, AnomalyCalibrator)
    assert calibrator.n_samples > 0


def test_raw_isolation_forest_scores_can_be_negative():
    iso = joblib.load("models/anomaly_model.pkl")
    # Decision function on extreme values can be negative
    extreme_sample = np.array([[999999.0] * iso.n_features_in_])
    score = iso.decision_function(extreme_sample)[0]
    assert score < 0, f"Expected negative raw Isolation Forest score, got {score}"


def test_calibrated_anomaly_signals_are_always_bounded_zero_to_one():
    test_scores = [-100.0, -10.0, -0.5, -0.2, 0.0, 0.1, 0.15, 0.2, 5.0, 100.0]
    for s in test_scores:
        norm = _normalize_anomaly_score(s)
        assert 0.0 <= norm <= 1.0, f"Calibrated anomaly out of bounds for {s}: {norm}"
        
    array_scores = np.array(test_scores)
    norm_arr = _normalize_anomaly_score(array_scores)
    assert np.all(norm_arr >= 0.0)
    assert np.all(norm_arr <= 1.0)


def test_normal_reference_observations_do_not_receive_arbitrary_floor():
    calibrator = get_anomaly_calibrator()
    normal_score = calibrator.max_score
    norm_signal = _normalize_anomaly_score(normal_score)
    # Must map close to 0.0 (not >= 0.30)
    assert norm_signal < 0.05, f"Expected low anomaly signal for normal inlier, got {norm_signal}"


def test_extreme_anomaly_values_map_toward_high_anomaly_signal():
    calibrator = get_anomaly_calibrator()
    extreme_anomaly = calibrator.min_score - 0.1
    norm_signal = _normalize_anomaly_score(extreme_anomaly)
    assert norm_signal >= 0.95, f"Expected high anomaly signal for extreme anomaly, got {norm_signal}"


def test_risk_remains_within_zero_to_one():
    for f_prob in [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0]:
        for anom in [-10.0, -0.3, 0.0, 0.15, 1.0, 10.0]:
            risk = calculate_risk(f_prob, anom)
            assert 0.0 <= risk <= 1.0, f"Risk out of bounds: {risk}"


def test_risk_calculation_uses_calibrated_anomaly_not_raw():
    # If calculate_risk used raw score (-0.20), 0.7 * 0.5 + 0.3 * (-0.20) = 0.29
    # With calibrated anomaly (-0.20 -> ~1.0), 0.7 * 0.5 + 0.3 * 1.0 = 0.65
    raw_anom = -0.20
    calibrated = _normalize_anomaly_score(raw_anom)
    assert calibrated > 0.90
    risk = calculate_risk(0.5, raw_anom)
    expected_risk = min(1.0, max(0.0, 0.7 * 0.5 + 0.3 * calibrated))
    assert np.isclose(risk, expected_risk, atol=1e-4)


def test_decision_engine_preserves_four_distinct_risk_bands():
    # Verify the four risk bands:
    # risk < 0.30       -> Normal
    # 0.30 <= risk < 0.50 -> Scale Resources
    # 0.50 <= risk < 0.65 -> Migrate VM
    # risk >= 0.65      -> Restart Task
    test_cases = [
        (0.10, "Normal"),
        (0.29, "Normal"),
        (0.30, "Scale Resources"),
        (0.45, "Scale Resources"),
        (0.50, "Migrate VM"),
        (0.60, "Migrate VM"),
        (0.65, "Restart Task"),
        (0.90, "Restart Task"),
    ]
    for r, expected in test_cases:
        base = get_base_action(r)
        decision = make_decision(f"test_{r}", r, 0.95, default_action_estimates(r))
        assert base == expected, f"Base action mismatch for risk {r}: expected {expected}, got {base}"
        assert decision["recommended_action"] == expected, f"Action mismatch for risk {r}: expected {expected}, got {decision['recommended_action']}"


def test_actions_are_not_hardcoded():
    actions_seen = set()
    for r in [0.15, 0.35, 0.55, 0.85]:
        decision = make_decision(f"dyn_{r}", r, 0.95, default_action_estimates(r))
        actions_seen.add(decision["recommended_action"])
    assert actions_seen == {"Normal", "Scale Resources", "Migrate VM", "Restart Task"}
