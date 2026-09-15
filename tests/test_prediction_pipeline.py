import json

import joblib

from src.pipeline import run_decision_pipeline
from src.preprocessing import preprocess_data


def test_saved_model_prediction_pipeline_returns_structured_decision():
    model = joblib.load("models/failure_model.pkl")
    anomaly_model = joblib.load("models/anomaly_model.pkl")
    data = preprocess_data("data/raw/borg_traces_data.csv", nrows=1000)
    features = data.drop(columns=["failed"]).reindex(columns=model.feature_names_in_, fill_value=0)
    decision = run_decision_pipeline(model, anomaly_model, features.iloc[[0]], "pipeline-test")
    assert 0 <= decision["risk"] <= 1
    assert decision["recommended_action"] in {"Normal", "Scale Resources", "Migrate VM", "Restart Task"}
    json.dumps(decision)
