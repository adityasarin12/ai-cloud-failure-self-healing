import joblib
import numpy as np
import pytest

from src.boundary_analysis import analyze_single_prediction_boundary
from src.decision_engine import default_action_estimates, make_decision
from src.single_prediction import build_single_prediction_features


@pytest.fixture(scope="module")
def saved_models():
    return (
        joblib.load("models/failure_model.pkl"),
        joblib.load("models/anomaly_model.pkl"),
        joblib.load("models/feature_medians.pkl"),
    )


def test_decision_engine_test_matrix_is_independent_of_ml(saved_models):
    matrix = [
        (0.00, "Normal"), (0.299, "Normal"),
        (0.300, "Scale Resources"), (0.499, "Scale Resources"),
        (0.500, "Migrate VM"), (0.649, "Migrate VM"),
        (0.650, "Restart Task"), (0.900, "Restart Task"),
    ]
    for risk, expected in matrix:
        assert make_decision("policy-boundary", risk, 0.95, default_action_estimates(risk))["recommended_action"] == expected


def test_single_prediction_payload_preserves_current_slider_values(saved_models):
    model, _, medians = saved_models
    values = {
        "priority": 321,
        "cpu_mean": 0.73,
        "assigned_memory": 0.64,
        "cycles_per_instruction": 1.4,
        "memory_accesses_per_instruction": 0.82,
        "sample_rate": 0.37,
    }
    features = build_single_prediction_features(values, model.feature_names_in_, medians)
    assert features.iloc[0][list(values)].to_dict() == values
    assert features.iloc[0]["avg_cpu"] == values["cpu_mean"]
    assert features.iloc[0]["avg_memory"] == values["assigned_memory"]


def test_single_prediction_boundary_is_reproducible_and_reports_all_fields(saved_models):
    results = analyze_single_prediction_boundary(*saved_models, random_count=200, seed=42)
    required = {
        "priority", "cpu_mean", "assigned_memory", "cycles_per_instruction",
        "memory_accesses_per_instruction", "sample_rate", "failure_probability",
        "raw_isolation_forest_score", "calibrated_anomaly_signal", "risk_score",
        "risk_band", "base_action", "final_action", "confidence",
    }
    assert required <= set(results.columns)
    assert results["risk_score"].between(0, 1).all()
    assert np.isfinite(results["risk_score"]).all()
    assert len(results) == 208


def test_boundary_analysis_demonstrates_reachable_action_limitation(saved_models):
    results = analyze_single_prediction_boundary(*saved_models, random_count=2000, seed=42)
    distribution = results["risk_band"].value_counts()
    assert set(distribution.index) <= {"Scale Resources", "Migrate VM"}
    assert results["risk_score"].min() >= 0.30
    assert results["risk_score"].max() < 0.65