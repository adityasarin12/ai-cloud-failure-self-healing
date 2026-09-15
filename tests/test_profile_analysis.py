import pandas as pd

from src.profile_analysis import (
    action_distribution,
    analyze_profile_actions,
    first_profile_per_action,
    risk_summary,
)


class _Model:
    feature_names_in_ = ("priority", "req_cpu", "avg_cpu", "assigned_memory")


def test_profile_action_analysis_uses_ui_builder_and_production_pipeline(monkeypatch):
    profiles = pd.DataFrame([
        {"priority": 1, "req_cpu": 0.1, "avg_cpu": 0.2, "assigned_memory": 0.3},
        {"priority": 2, "req_cpu": 0.4, "avg_cpu": 0.5, "assigned_memory": 0.6},
    ])
    calls = []

    def fake_pipeline(model, anomaly_model, features, incident_id):
        calls.append((list(features.columns), features.iloc[0].to_dict(), incident_id))
        return {
            "failure_probability": 0.2,
            "raw_anomaly_score": 0.1,
            "calibrated_anomaly": 0.3,
            "risk": 0.23,
            "recommended_action": "Normal",
            "confidence": 0.9,
            "confidence_level": "HIGH",
        }

    monkeypatch.setattr("src.profile_analysis.run_decision_pipeline", fake_pipeline)
    results = analyze_profile_actions(_Model(), object(), profiles)

    assert len(calls) == len(profiles)
    assert calls[0][0] == list(_Model.feature_names_in_)
    assert calls[0][1] == profiles.iloc[0].to_dict()
    assert calls[1][2] == "reference-profile-1"
    assert results["final_recommended_action"].tolist() == ["Normal", "Normal"]


def test_profile_action_statistics_include_all_actions_and_first_real_profile():
    results = pd.DataFrame({
        "profile_index": [3, 9, 12],
        "risk_score": [0.1, 0.4, 0.8],
        "final_recommended_action": ["Normal", "Scale Resources", "Restart Task"],
    })
    distribution = action_distribution(results)
    assert distribution["Action"].tolist() == ["Normal", "Scale Resources", "Migrate VM", "Restart Task"]
    assert distribution["Count"].tolist() == [1, 1, 0, 1]
    assert risk_summary(results).loc[0.0] == 0.1
    representatives = first_profile_per_action(results)
    assert set(representatives["profile_index"]) == {3, 9, 12}
