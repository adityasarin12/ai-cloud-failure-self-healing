import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt
import time
from src.preprocessing import preprocess_data
from src.predict import generate_results
from src.pipeline import run_decision_pipeline
from src.profile_prediction import (
    IMPORTANT_TELEMETRY_FEATURES,
    build_profile_prediction_features,
    load_reference_profiles,
    manual_telemetry_specs,
    input_diagnostics,
    telemetry_distribution_audit,
    profile_distribution_diagnostic,
)
from src.profile_analysis import (
    action_distribution,
    analyze_profile_actions,
    write_profile_action_analysis,
)
from src.forecasting import (
    forecast_cpu_memory,
    forecast_status,
    load_ordered_load_series,
    proactive_recommendation,
)
from config import DATA_PATH

PROFILE_ANALYSIS_PATH = "data/processed/single_prediction_profile_action_analysis.csv"

st.set_page_config(page_title="AI Cloud System", layout="wide")

st.title("AI-Driven Cloud Failure Prediction System")

# Load models
@st.cache_resource
def load_models():
    model = joblib.load("models/failure_model.pkl")
    iso = joblib.load("models/anomaly_model.pkl")
    return model, iso

model, iso = load_models()


@st.cache_data(show_spinner="Loading realistic training profiles...")
def load_profiles(data_path, model_features):
    """Load a cached real-profile reference set for submit-only predictions."""
    return load_reference_profiles(data_path, model_features)


@st.cache_data(show_spinner="Preparing batch analysis...")
def load_batch_results(data_path, _model, _iso):
    """Cache expensive batch work so unrelated UI reruns stay responsive."""
    df = preprocess_data(data_path)
    return generate_results(_model, _iso, df.drop("failed", axis=1))


@st.cache_data(show_spinner="Preparing independent load forecast...")
def load_forecast_series(data_path):
    return load_ordered_load_series(data_path)


@st.cache_data(show_spinner="Analyzing real reference-profile actions...")
def load_profile_action_analysis(output_path, _model, _iso, profiles):
    """Use saved real-profile evidence when it matches the current reference set."""
    try:
        existing = pd.read_csv(output_path)
        if len(existing) == len(profiles):
            return existing
    except FileNotFoundError:
        pass
    results = analyze_profile_actions(_model, _iso, profiles)
    write_profile_action_analysis(results, output_path)
    return results


if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "batch_analysis_requested" not in st.session_state:
    st.session_state.batch_analysis_requested = False

# -------------------- TABS --------------------
tab1, tab2 = st.tabs(["Single Prediction", "Batch Analysis"])

# =========================================================
# 🔹 TAB 1 — SINGLE PREDICTION
# =========================================================
with tab1:

    st.subheader("Real-Time Task Prediction")

    profiles = load_profiles(DATA_PATH, tuple(model.feature_names_in_))
    source_mode = st.radio(
        "Prediction Source",
        ("Realistic Training Profile", "Manual Telemetry"),
        horizontal=True,
        help="Profiles are real preprocessed Borg task rows in the model's exact feature order.",
    )
    selected_index = st.selectbox(
        "Workload Profile",
        options=list(range(len(profiles))),
        format_func=lambda index: f"Profile #{index} — Reference Borg Task",
        help="Select an actual preprocessed Borg reference task.",
    )
    selected_profile = profiles.iloc[int(selected_index)].copy()
    profile_id = selected_profile["Unnamed: 0"]
    metadata_col1, metadata_col2, metadata_col3 = st.columns(3)
    metadata_col1.metric("Profile ID", str(int(profile_id)))
    metadata_col2.metric("Dataset / Source", "Borg reference")
    metadata_col3.metric("Model Features", len(model.feature_names_in_))
    telemetry_columns = [feature for feature in IMPORTANT_TELEMETRY_FEATURES if feature in profiles]
    st.caption("Selected profile telemetry (unchanged unless Manual Telemetry edits are submitted).")
    st.dataframe(
        selected_profile[telemetry_columns].rename("Selected value").to_frame(),
        use_container_width=True,
    )

    with st.form("prediction_form"):
        edits = {}
        if source_mode == "Manual Telemetry":
            st.caption("Each field is independent; bounds are the 1st--99th reference percentiles.")
            specs = manual_telemetry_specs(profiles, selected_profile)
            columns = st.columns(3)
            for index, feature in enumerate(telemetry_columns):
                spec = specs[feature]
                edits[feature] = columns[index % 3].slider(
                    feature.replace("_", " ").title(),
                    min_value=spec["min"], max_value=spec["max"], value=spec["default"],
                    step=spec["step"],
                    format=spec["format"],
                    key=f"manual_{feature}",
                )
        predict_clicked = st.form_submit_button("Predict", type="primary")

    if predict_clicked:
        input_data = build_profile_prediction_features(
            selected_profile,
            model.feature_names_in_,
            edits=edits if source_mode == "Manual Telemetry" else None,
        )

        with st.spinner("Predicting..."):
            decision = run_decision_pipeline(model, iso, input_data, "interactive_task")
            failure_prob = decision["failure_probability"]
            calibrated_anomaly = decision["anomaly_score"]
            raw_anomaly = decision.get("raw_anomaly_score", 0.0)
            risk = decision["risk"]

            forecast_started = time.perf_counter()
            forecast = forecast_cpu_memory(load_forecast_series(DATA_PATH), horizon=3)
            forecast["status"] = forecast_status(forecast)
            forecast["proactive_recommendation"] = proactive_recommendation(
                decision["recommended_action"], forecast
            )
            decision["forecast"] = forecast
            decision["latency"]["forecast_ms"] = (time.perf_counter() - forecast_started) * 1000
            decision["latency"]["total_inference_ms"] += decision["latency"]["forecast_ms"]

            st.session_state.prediction_history.append({
                "prediction": len(st.session_state.prediction_history) + 1,
                "failure_probability": failure_prob,
                "anomaly_score": calibrated_anomaly,
                "raw_anomaly_score": raw_anomaly,
                "anomaly_signal": calibrated_anomaly,
                "risk_score": risk,
            })

        st.markdown("---")
        st.subheader("Prediction Results")

        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)

        c1.metric("Failure Probability", round(failure_prob, 3))
        c2.metric("Anomaly Score", round(calibrated_anomaly, 3))
        c3.metric("Risk Score", round(risk, 3))
        c4.metric("Confidence", round(decision["confidence"], 3))
        c5.metric("Recommended Action", decision["recommended_action"])
        c6.metric("Confidence Level", decision["confidence_level"])
        st.caption(
            f"Prediction Source: {source_mode} · Profile ID: {int(profile_id)}"
        )

        with st.expander("Structured Decision (AWS Integration)"):
            st.json(decision)

        with st.expander("Why this decision?"):
            st.write(f"**Risk Band:** {decision['risk_band']}")
            st.write(f"**Selected Action:** {decision['recommended_action']}")
            st.write(f"**Why this action was selected:** {decision['reason']}")
            st.write(f"**Confidence Level:** {decision['confidence_level']}")
            st.write(f"**Raw Isolation Forest Score:** `{round(raw_anomaly, 4)}`")
            st.write(f"**Calibrated Anomaly Signal:** `{round(calibrated_anomaly, 4)}`")
            st.write(f"Policy status: {decision['policy_status']}")
            st.write(f"Rollback required: {'Yes' if decision['rollback_required'] else 'No'}")
            diagnosis = decision.get("diagnosis", {})
            st.write(diagnosis.get("warning", "No diagnostic explanation available."))
            if diagnosis.get("drivers"):
                st.dataframe(pd.DataFrame(diagnosis["drivers"]), hide_index=True, use_container_width=True)
            explanation = decision.get("explanation", {})
            st.write(explanation.get("primary_reason", "No model-based explanation available."))
            st.caption(explanation.get("disclaimer", "Model-based explanation — not causal proof."))
            factors = explanation.get("contributing_factors", [])
            if factors:
                st.dataframe(pd.DataFrame(factors)[["feature", "value", "impact", "direction"]], hide_index=True, use_container_width=True)

        with st.expander("Performance"):
            latency = decision.get("latency", {})
            st.metric("Total Inference Latency", f"{latency.get('total_inference_ms', 0.0):.2f} ms")
            st.write({
                "RF Latency": f"{latency.get('failure_model_ms', 0.0):.2f} ms",
                "Anomaly Latency": f"{latency.get('anomaly_model_ms', 0.0):.2f} ms",
                "RCA Latency": f"{latency.get('explanation_ms', 0.0):.2f} ms",
                "Forecast Latency": f"{latency.get('forecast_ms', 0.0):.2f} ms",
            })

        with st.expander("Load Forecast"):
            st.write(f"Current CPU: {forecast['current_cpu']:.6g}")
            st.write("Forecast CPU:", [f"t+{i + 1} = {value:.6g}" for i, value in enumerate(forecast["forecast_cpu"])])
            st.write(f"Current Memory: {forecast['current_memory']:.6g}")
            st.write("Forecast Memory:", [f"t+{i + 1} = {value:.6g}" for i, value in enumerate(forecast["forecast_memory"])])
            st.write(f"Forecast Trend: {forecast['status']}")
            st.write(f"Forecast-based Recommendation: {forecast['proactive_recommendation']}")

        with st.expander("🔍 Input Distribution Diagnostics", expanded=False):
            diagnostic, out_of_distribution = input_diagnostics(input_data.iloc[0], profiles)
            in_range = len(diagnostic) - len(out_of_distribution)
            st.write(f"{len(model.feature_names_in_)} model features · {in_range} in range · {len(out_of_distribution)} OOD")
            st.caption(
                "Reference ranges are calculated from the training/reference workload distribution. "
                "OOD indicates that the current telemetry is outside the central 98% reference range."
            )
            display = diagnostic.copy()
            display["Status"] = display["distribution_status"]
            display["Feature"] = display["feature"]
            display["Current"] = display["profile_value"]
            display["P01"] = display["reference_p01"]
            display["Median"] = display["reference_median"]
            display["P99"] = display["reference_p99"]
            display = display.sort_values("Status", key=lambda values: values.map({"LOW OOD": 0, "HIGH OOD": 0, "IN RANGE": 1}))
            for column in ("Current", "P01", "Median", "P99"):
                display[column] = display[column].map(
                    lambda value: str(int(value)) if float(value).is_integer() and abs(float(value)) >= 1 else f"{float(value):.6g}".rstrip("0").rstrip(".")
                )
            st.dataframe(display[["Feature", "Current", "P01", "Median", "P99", "Status"]], hide_index=True, use_container_width=True)

    history = pd.DataFrame(st.session_state.prediction_history)
    if not history.empty:
        st.markdown("---")
        st.subheader("System Risk Timeline")
        timeline = history[[
            "prediction", "risk_score", "failure_probability", "anomaly_signal"
        ]].rename(columns={
            "prediction": "Prediction",
            "risk_score": "Risk Score",
            "failure_probability": "Failure Probability",
            "anomaly_signal": "Anomaly Score",
        }).melt(
            id_vars="Prediction",
            var_name="Metric",
            value_name="Score",
        )
        chart = alt.Chart(timeline).mark_line(point=True).encode(
            x=alt.X("Prediction:Q", title="Prediction Sequence"),
            y=alt.Y("Score:Q", title="Normalized Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=["Prediction:Q", "Metric:N", alt.Tooltip("Score:Q", format=".3f")],
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        st.caption("Anomaly signal is calibrated to [0, 1] range; the raw Isolation Forest score is preserved in the structured decision.")

# =========================================================
# 🔹 TAB 2 — BATCH ANALYSIS
# =========================================================
with tab2:

    st.subheader("Dataset Analysis")

    if st.button("Run Batch Analysis", type="primary"):
        st.session_state.batch_analysis_requested = True

    if not st.session_state.batch_analysis_requested:
        st.info("Run batch analysis when you are ready to inspect the dataset.")
    else:
        results = load_batch_results(DATA_PATH, model, iso)

        # KPIs
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Tasks", len(results))
        col2.metric("High Risk Tasks", (results["risk_score"] > 0.6).sum())
        col3.metric("Avg Risk", round(results["risk_score"].mean(), 3))

        st.markdown("---")

        # Action Distribution
        st.subheader("Action Distribution")
        st.bar_chart(results["action"].value_counts())

        st.subheader("Reference Profile Action Distribution")
        profile_actions = load_profile_action_analysis(
            PROFILE_ANALYSIS_PATH, model, iso, profiles
        )
        st.dataframe(action_distribution(profile_actions), hide_index=True, use_container_width=True)

        # Risk Distribution
        st.subheader("Risk Score Distribution")
        st.bar_chart(results["risk_score"].value_counts(bins=20))

        # Top Risky Tasks
        st.subheader("Top 10 High Risk Tasks")
        top_risk = results.sort_values("risk_score", ascending=False).head(10)
        st.dataframe(top_risk)

        # Filter
        st.subheader("Filter by Risk Threshold")
        threshold = st.slider("Select threshold", 0.0, 1.0, 0.5)

        filtered = results[results["risk_score"] > threshold]
        st.write(f"Tasks above threshold: {len(filtered)}")

        st.dataframe(filtered.head(20))
