import joblib
import numpy as np
import pandas as pd

from src.pipeline import run_decision_pipeline
from src.profile_prediction import (
    build_profile_prediction_features,
    load_reference_profiles,
    manual_telemetry_specs,
    profile_distribution_diagnostic,
    input_diagnostics,
    telemetry_distribution_audit,
)
from src.profile_analysis import analyze_manual_telemetry


def test_selected_reference_profile_has_all_model_features_in_exact_order():
    model = joblib.load("models/failure_model.pkl")
    profiles = load_reference_profiles("data/raw/borg_traces_data.csv", model.feature_names_in_, nrows=250)
    row = build_profile_prediction_features(profiles.iloc[7], model.feature_names_in_)
    assert len(profiles.columns) == model.n_features_in_ == 26
    assert list(row.columns) == list(model.feature_names_in_)


def test_profile_values_are_unchanged_without_edits_and_not_synthetically_equalized():
    model = joblib.load("models/failure_model.pkl")
    profiles = load_reference_profiles("data/raw/borg_traces_data.csv", model.feature_names_in_, nrows=250)
    profile = next(
        candidate for _, candidate in profiles.iterrows()
        if len(set(candidate[["req_cpu", "avg_cpu", "max_cpu", "sample_cpu", "cpu_mean", "tail_cpu_mean"]])) > 1
        and len(set(candidate[["assigned_memory", "req_memory", "avg_memory", "max_memory"]])) > 1
    )
    row = build_profile_prediction_features(profile, model.feature_names_in_).iloc[0]
    assert row.to_dict() == profile.loc[list(model.feature_names_in_)].to_dict()
    assert len(set(row[["req_cpu", "avg_cpu", "max_cpu", "sample_cpu", "cpu_mean", "tail_cpu_mean"]])) > 1
    assert len(set(row[["assigned_memory", "req_memory", "avg_memory", "max_memory"]])) > 1


def test_selected_profile_prediction_keeps_calibration_and_risk_bounded():
    model = joblib.load("models/failure_model.pkl")
    anomaly_model = joblib.load("models/anomaly_model.pkl")
    profiles = load_reference_profiles("data/raw/borg_traces_data.csv", model.feature_names_in_, nrows=250)
    row = build_profile_prediction_features(profiles.iloc[12], model.feature_names_in_)
    decision = run_decision_pipeline(model, anomaly_model, row, "selected-profile-test")
    assert 0.0 <= decision["anomaly_score"] <= 1.0
    assert 0.0 <= decision["risk"] <= 1.0


def test_profile_distribution_diagnostic_reports_every_model_feature_in_range():
    model = joblib.load("models/failure_model.pkl")
    profiles = load_reference_profiles("data/raw/borg_traces_data.csv", model.feature_names_in_, nrows=250)
    diagnostic = profile_distribution_diagnostic(profiles.iloc[24], profiles)
    assert list(diagnostic["feature"]) == list(model.feature_names_in_)
    assert diagnostic["within_reference_distribution"].all()


def test_manual_telemetry_specs_use_quantiles_and_preserve_selected_defaults():
    model = joblib.load("models/failure_model.pkl")
    profiles = load_reference_profiles("data/raw/borg_traces_data.csv", model.feature_names_in_, nrows=5000)
    selected = profiles.iloc[7]
    specs = manual_telemetry_specs(profiles, selected)
    for feature in manual_telemetry_specs(profiles):
        values = profiles[feature]
        assert specs[feature]["min"] == float(values.quantile(0.01))
        assert specs[feature]["max"] == float(values.quantile(0.99))
        assert specs[feature]["min"] < specs[feature]["max"]
        assert specs[feature]["min"] <= specs[feature]["default"] <= specs[feature]["max"]
        assert specs[feature]["step"] > 0


def test_page_cache_edit_changes_only_page_cache_memory_and_is_independent_of_assigned_memory():
    model = joblib.load("models/failure_model.pkl")
    profiles = load_reference_profiles("data/raw/borg_traces_data.csv", model.feature_names_in_, nrows=5000)
    profile = profiles.iloc[7]
    specs = manual_telemetry_specs(profiles, profile)
    edited_value = specs["page_cache_memory"]["max"]
    original = build_profile_prediction_features(profile, model.feature_names_in_).iloc[0]
    edited = build_profile_prediction_features(
        profile, model.feature_names_in_, {"page_cache_memory": edited_value}
    ).iloc[0]
    changed = [feature for feature in model.feature_names_in_ if original[feature] != edited[feature]]
    assert changed == ["page_cache_memory"]
    assert edited["page_cache_memory"] != edited["assigned_memory"]


def test_page_cache_slider_has_visible_precision_and_selected_value_propagates():
    model = joblib.load("models/failure_model.pkl")
    profiles = load_reference_profiles("data/raw/borg_traces_data.csv", model.feature_names_in_, nrows=5000)
    specs = manual_telemetry_specs(profiles, profiles.iloc[7])
    page_cache = specs["page_cache_memory"]
    assert page_cache["min"] < page_cache["max"]
    assert int(page_cache["format"].split(".")[1][:-1]) >= 5
    assert format(page_cache["min"], page_cache["format"][1:]) != format(page_cache["max"], page_cache["format"][1:])
    row = build_profile_prediction_features(profiles.iloc[7], model.feature_names_in_)
    assert row.iloc[0]["page_cache_memory"] == profiles.iloc[7]["page_cache_memory"]


def test_manual_edits_do_not_artificially_couple_cpu_or_memory_features():
    model = joblib.load("models/failure_model.pkl")
    profiles = load_reference_profiles("data/raw/borg_traces_data.csv", model.feature_names_in_, nrows=250)
    profile = profiles.iloc[7]
    original = build_profile_prediction_features(profile, model.feature_names_in_).iloc[0]
    edited = build_profile_prediction_features(
        profile, model.feature_names_in_, {"cpu_mean": 0.123, "assigned_memory": 0.045}
    ).iloc[0]
    assert edited["cpu_mean"] == 0.123
    assert edited["assigned_memory"] == 0.045
    for feature in ("req_cpu", "avg_cpu", "max_cpu", "sample_cpu", "tail_cpu_mean", "req_memory", "avg_memory", "max_memory"):
        assert edited[feature] == original[feature]


def test_input_diagnostics_identifies_out_of_distribution_features():
    model = joblib.load("models/failure_model.pkl")
    profiles = load_reference_profiles("data/raw/borg_traces_data.csv", model.feature_names_in_, nrows=250)
    row = profiles.iloc[0].copy()
    row["priority"] = profiles["priority"].max() + 1
    diagnostic, outside = input_diagnostics(row, profiles)
    assert diagnostic.loc[diagnostic["feature"] == "priority", "distribution_status"].item() == "HIGH OOD"
    assert "priority" in outside


def test_telemetry_distribution_audit_reports_training_and_ui_ranges():
    model = joblib.load("models/failure_model.pkl")
    profiles = load_reference_profiles("data/raw/borg_traces_data.csv", model.feature_names_in_, nrows=250)
    audit = telemetry_distribution_audit(profiles, profiles.iloc[0])
    assert len(audit) == len(manual_telemetry_specs(profiles))
    assert {"training_min", "training_p01", "training_median", "training_p99", "training_max", "ui_min", "ui_max", "ui_default", "status"} <= set(audit.columns)
    assert (audit["ui_min"] < audit["ui_max"]).all()
    assert (audit["status"] == "IN_DISTRIBUTION").all()


def test_diagnostic_status_uses_central_98_percent_reference_range():
    reference = pd.DataFrame({"telemetry": [0.0, 1.0, 2.0, 3.0, 100.0]})
    low = profile_distribution_diagnostic(pd.Series({"telemetry": -1.0}), reference)
    high = profile_distribution_diagnostic(pd.Series({"telemetry": 101.0}), reference)
    assert low.iloc[0]["distribution_status"] == "LOW OOD"
    assert high.iloc[0]["distribution_status"] == "HIGH OOD"


def test_manual_and_real_profile_audits_return_action_and_risk_metrics():
    model = joblib.load("models/failure_model.pkl")
    anomaly_model = joblib.load("models/anomaly_model.pkl")
    profiles = load_reference_profiles("data/raw/borg_traces_data.csv", model.feature_names_in_, nrows=250)
    results = analyze_manual_telemetry(model, anomaly_model, profiles, sample_count=10)
    assert len(results) == 13
    assert results["risk_score"].between(0, 1).all()
    assert results["action"].isin({"Normal", "Scale Resources", "Migrate VM", "Restart Task"}).all()
