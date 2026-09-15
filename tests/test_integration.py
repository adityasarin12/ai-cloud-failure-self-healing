import numpy as np

import joblib

from src.predict import generate_results
from src.preprocessing import preprocess_data


def test_batch_analysis_has_valid_decisions_without_retraining():
    model = joblib.load("models/failure_model.pkl")
    anomaly_model = joblib.load("models/anomaly_model.pkl")
    data = preprocess_data("data/raw/borg_traces_data.csv", nrows=1000)
    features = data.drop(columns=["failed"]).reindex(columns=model.feature_names_in_, fill_value=0)
    results = generate_results(model, anomaly_model, features)
    allowed = {"Normal", "Scale Resources", "Migrate VM", "Restart Task"}
    assert results["action"].isin(allowed).all()
    assert results["risk_score"].between(0, 1).all()
    assert np.isfinite(results[["risk_score", "failure_probability", "confidence_score"]].to_numpy()).all()
