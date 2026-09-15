import numpy as np

from src.risk_score import calculate_risk


def test_risk_is_bounded_for_valid_inputs():
    for value in (0, 0.25, 0.5, 0.75, 1):
        risk = calculate_risk(value, 0.0)
        assert 0 <= risk <= 1


def test_risk_clips_out_of_range_failure_probability():
    assert calculate_risk(-1, 0.5) == 0
    assert calculate_risk(2, 0.5) == 1


def test_negative_anomaly_score_cannot_make_risk_invalid():
    risk = calculate_risk(0.8, -10)
    assert 0 <= risk <= 1
