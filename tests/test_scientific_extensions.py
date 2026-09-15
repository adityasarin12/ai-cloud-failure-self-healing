import json

import joblib
import numpy as np
import pandas as pd

from src.diagnosis import explain_prediction
from src.evaluate import evaluate_classifier, summarize_latency
from src.forecasting import (
    evaluate_temporal_forecast,
    forecast_cpu_memory,
    forecast_status,
    temporal_split,
)
from src.pipeline import run_decision_pipeline
from src.preprocessing import preprocess_data


def _saved_inputs():
    model = joblib.load("models/failure_model.pkl")
    anomaly = joblib.load("models/anomaly_model.pkl")
    data = preprocess_data("data/raw/borg_traces_data.csv", nrows=100)
    return model, anomaly, data.drop(columns=["failed"]).reindex(columns=model.feature_names_in_, fill_value=0)


def test_model_explanation_uses_real_features_and_disclaims_causality():
    model, anomaly, features = _saved_inputs()
    row = features.iloc[0]
    explanation = explain_prediction(model, row.to_numpy(), anomaly_model=anomaly, feature_names=features.columns, top_k=3)
    assert len(explanation["contributing_factors"]) == 3
    assert all(item["feature"] in model.feature_names_in_ for item in explanation["contributing_factors"])
    assert "not causal" in explanation["disclaimer"].lower()
    json.dumps(explanation)


def test_pipeline_exposes_latency_and_explanation_without_changing_risk_contract():
    model, anomaly, features = _saved_inputs()
    decision = run_decision_pipeline(model, anomaly, features.iloc[[0]], "extension-test")
    assert 0 <= decision["risk"] <= 1
    assert decision["explanation"]["recommended_action"] == decision["recommended_action"]
    assert all(value >= 0 for value in decision["latency"].values())
    assert decision["latency"]["total_inference_ms"] >= decision["latency"]["failure_model_ms"]


def test_temporal_split_has_no_future_leakage_and_forecast_shape():
    series = pd.DataFrame({
        "start_time": np.arange(20),
        "cpu_load": np.linspace(0.1, 0.5, 20),
        "memory_load": np.linspace(0.2, 0.6, 20),
    })
    train, test = temporal_split(series, holdout_fraction=0.2)
    assert train.start_time.max() < test.start_time.min()
    forecast = forecast_cpu_memory(series, horizon=3)
    assert len(forecast["forecast_cpu"]) == len(forecast["forecast_memory"]) == 3
    assert all(value >= 0 for value in forecast["forecast_cpu"] + forecast["forecast_memory"])
    assert forecast_status(forecast) in {"Stable", "Increasing", "High load expected", "Critical load expected"}


def test_temporal_forecast_reports_linear_and_persistence_metrics():
    series = pd.DataFrame({
        "start_time": np.arange(30),
        "cpu_load": np.linspace(0.1, 0.8, 30),
        "memory_load": np.linspace(0.2, 0.7, 30),
    })
    result = evaluate_temporal_forecast(series)
    assert result["split_is_chronological"]
    assert {"mae", "rmse", "mape"} <= set(result["metrics"]["cpu_load"]["linear"])


def test_classifier_metrics_include_required_probability_metrics():
    model, _, features = _saved_inputs()
    y = np.array([0, 1] * (len(features) // 2))
    metrics = evaluate_classifier(model, features, y)
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"} <= set(metrics)


def test_latency_summary_has_required_statistics():
    summary = summarize_latency(np.array([1.0, 2.0, 3.0]))
    assert {"mean", "median", "p50", "p95", "p99", "min", "max", "std"} <= set(summary)