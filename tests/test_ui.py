import pytest
import numpy as np
from pathlib import Path


try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None


@pytest.mark.skipif(AppTest is None, reason="Streamlit AppTest is unavailable")
def test_streamlit_app_starts_with_required_controls():
    app = AppTest.from_file("app.py").run(timeout=10)
    assert not app.exception
    assert {tab.label for tab in app.tabs} >= {"Single Prediction", "Batch Analysis"}
    assert any(item.label == "Prediction Source" for item in app.radio)
    assert any(item.label == "Workload Profile" for item in app.selectbox)
    assert any(button.label == "Predict" for button in app.button)


@pytest.mark.skipif(AppTest is None, reason="Streamlit AppTest is unavailable")
def test_streamlit_prediction_displays_clean_decision_fields():
    app = AppTest.from_file("app.py").run(timeout=10)
    next(button for button in app.button if button.label == "Predict").click().run(timeout=10)
    assert not app.exception
    labels = {element.label for element in app.metric}
    assert {"Failure Probability", "Anomaly Score", "Risk Score", "Confidence", "Recommended Action", "Confidence Level"} <= labels
    assert all(item.value in {"Normal", "Scale Resources", "Migrate VM", "Restart Task"} for item in app.metric if item.label == "Recommended Action")
    
    # Verify displayed Anomaly Score is strictly non-negative and in [0, 1]
    anomaly_metric = next(item.value for item in app.metric if item.label == "Anomaly Score")
    anom_val = float(anomaly_metric)
    assert 0.0 <= anom_val <= 1.0, f"Displayed anomaly score is negative or out of bounds: {anom_val}"
    
    # Verify displayed Risk Score is mathematically consistent with displayed components
    fail_metric = float(next(item.value for item in app.metric if item.label == "Failure Probability"))
    risk_metric = float(next(item.value for item in app.metric if item.label == "Risk Score"))
    expected_risk = round(0.7 * fail_metric + 0.3 * anom_val, 3)
    assert np.isclose(risk_metric, expected_risk, atol=0.01), f"Displayed risk {risk_metric} does not match 0.7*{fail_metric} + 0.3*{anom_val} = {expected_risk}"


@pytest.mark.parametrize(
    ("profile_index", "expected_action"),
    [(2, "Normal"), (330, "Scale Resources"), (194, "Migrate VM"), (0, "Restart Task")],
)
@pytest.mark.skipif(AppTest is None, reason="Streamlit AppTest is unavailable")
def test_streamlit_real_profiles_display_nonempty_runtime_reason(profile_index, expected_action):
    app = AppTest.from_file("app.py").run(timeout=10)
    profile = next(item for item in app.selectbox if item.label == "Workload Profile")
    profile.set_value(profile_index).run(timeout=10)
    next(button for button in app.button if button.label == "Predict").click().run(timeout=10)
    assert not app.exception
    action = next(item.value for item in app.metric if item.label == "Recommended Action")
    assert action == expected_action
    rendered_text = " ".join(str(item.value) for item in app.markdown)
    assert "Why this action was selected:" in rendered_text
    assert "Failure probability" in rendered_text


@pytest.mark.skipif(AppTest is None, reason="Streamlit AppTest is unavailable")
def test_profile_changes_do_not_predict_until_submit():
    app = AppTest.from_file("app.py").run(timeout=10)
    assert not any(item.label == "Failure Probability" for item in app.metric)


@pytest.mark.skipif(AppTest is None, reason="Streamlit AppTest is unavailable")
def test_manual_page_cache_slider_accepts_changed_value():
    app = AppTest.from_file("app.py").run(timeout=10)
    source = next(item for item in app.radio if item.label == "Prediction Source")
    source.set_value("Manual Telemetry").run(timeout=10)
    slider = next(item for item in app.slider if item.label == "Page Cache Memory")
    assert slider.min < slider.max
    changed_value = slider.max - (slider.max - slider.min) / 2
    slider.set_value(changed_value).run(timeout=10)
    assert not app.exception
    assert next(item for item in app.slider if item.label == "Page Cache Memory").value == changed_value
    profile = next(item for item in app.selectbox if item.label == "Workload Profile")
    profile.set_value(1).run(timeout=10)
    assert not any(item.label == "Failure Probability" for item in app.metric)


@pytest.mark.skipif(AppTest is None, reason="Streamlit AppTest is unavailable")
def test_input_distribution_diagnostics_are_collapsed_and_compact():
    app = AppTest.from_file("app.py").run(timeout=10)
    next(button for button in app.button if button.label == "Predict").click().run(timeout=10)
    diagnostic = next(item for item in app.expander if item.label == "🔍 Input Distribution Diagnostics")
    assert 'with st.expander("🔍 Input Distribution Diagnostics", expanded=False)' in Path("app.py").read_text(encoding="utf-8")
    rendered = " ".join(str(item.value) for item in app.markdown)
    assert "Current:" not in rendered
    assert "Reference p01:" not in rendered
    assert not app.exception
    assert any(child.__class__.__name__ == "Dataframe" for child in diagnostic.children.values())
    assert any("26 model features" in str(item.value) for item in app.markdown)


@pytest.mark.skipif(AppTest is None, reason="Streamlit AppTest is unavailable")
def test_batch_analysis_runs_on_explicit_request():
    app = AppTest.from_file("app.py").run(timeout=10)
    next(button for button in app.button if button.label == "Run Batch Analysis").click().run(timeout=60)
    assert not app.exception
    labels = {item.label for item in app.metric}
    assert {"Total Tasks", "High Risk Tasks", "Avg Risk"} <= labels
