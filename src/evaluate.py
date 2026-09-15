import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
	accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
	f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
	roc_curve,
)
from sklearn.calibration import calibration_curve


def evaluate_classifier(model, X_test, y_test):
	probabilities = model.predict_proba(X_test)[:, 1]
	predictions = (probabilities >= 0.5).astype(int)
	return {
		"accuracy": float(accuracy_score(y_test, predictions)),
		"precision": float(precision_score(y_test, predictions, zero_division=0)),
		"recall": float(recall_score(y_test, predictions, zero_division=0)),
		"f1": float(f1_score(y_test, predictions, zero_division=0)),
		"roc_auc": float(roc_auc_score(y_test, probabilities)),
		"pr_auc": float(average_precision_score(y_test, probabilities)),
		"brier_score": float(brier_score_loss(y_test, probabilities)),
	}


def benchmark_prediction(model, iso, X, repetitions=3):
	timings = []
	for _ in range(repetitions):
		started = time.perf_counter()
		model.predict_proba(X)[:, 1]
		iso.decision_function(X)
		timings.append(time.perf_counter() - started)
	return {
		"rows": len(X),
		"mean_inference_seconds": sum(timings) / len(timings),
		"p95_inference_seconds": sorted(timings)[min(len(timings) - 1, int(len(timings) * 0.95))],
	}


def benchmark_components(model, iso, X, repetitions=1, explanation_fn=None):
	"""Measure inference components without including Streamlit rendering."""
	from src.risk_score import calculate_risk, _normalize_anomaly_score
 
	rows = []
	for run_id in range(repetitions):
		started = time.perf_counter()
		rf_started = time.perf_counter()
		probabilities = model.predict_proba(X)[:, 1]
		rf_ms = (time.perf_counter() - rf_started) * 1000
		iso_started = time.perf_counter()
		raw = iso.decision_function(X)
		isolation_ms = (time.perf_counter() - iso_started) * 1000
		calibration_started = time.perf_counter()
		anomaly = _normalize_anomaly_score(raw)
		calibration_ms = (time.perf_counter() - calibration_started) * 1000
		risk_started = time.perf_counter()
		risk = calculate_risk(probabilities, raw)
		risk_ms = (time.perf_counter() - risk_started) * 1000
		decision_started = time.perf_counter()
		from src.policy import get_base_action
		_ = [get_base_action(value) for value in risk]
		decision_ms = (time.perf_counter() - decision_started) * 1000
		rca_ms = 0.0
		if explanation_fn is not None:
			rca_started = time.perf_counter()
			for index in range(len(X)):
				explanation_fn(X.iloc[[index]] if hasattr(X, "iloc") else X[index:index + 1])
			rca_ms = (time.perf_counter() - rca_started) * 1000
		total_ms = (time.perf_counter() - started) * 1000
		rows.append({
			"run_id": run_id,
			"preprocessing_ms": 0.0,
			"rf_ms": rf_ms,
			"isolation_forest_ms": isolation_ms,
			"calibration_ms": calibration_ms,
			"risk_ms": risk_ms,
			"decision_ms": decision_ms,
			"rca_ms": rca_ms,
			"forecast_ms": 0.0,
			"total_ms": total_ms,
		})
	return pd.DataFrame(rows)


def summarize_latency(timings):
	"""Return reproducible latency summary statistics in milliseconds."""
	values = np.asarray(timings, dtype=float)
	return {
		"mean": float(values.mean()), "median": float(np.median(values)),
		"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
		"p99": float(np.percentile(values, 99)), "min": float(values.min()),
		"max": float(values.max()), "std": float(values.std()),
	}


def latency_summary_table(timings):
	"""Summarize every numeric latency component for paper output."""
	return pd.DataFrame({
		"component": column,
		**summarize_latency(timings[column].to_numpy()),
	} for column in timings.columns if column.endswith("_ms") or column.endswith("_ms"))


def paper_evaluation(model, X_test, y_test, results, output_dir="results"):
	"""Write metrics and paper figures from actual held-out predictions/results."""
	from pathlib import Path
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	root = Path(output_dir)
	metrics_dir, figures_dir = root / "metrics", root / "figures"
	metrics_dir.mkdir(parents=True, exist_ok=True)
	figures_dir.mkdir(parents=True, exist_ok=True)
	probabilities = model.predict_proba(X_test)[:, 1]
	predictions = (probabilities >= 0.5).astype(int)
	metrics = evaluate_classifier(model, X_test, y_test)
	pd.DataFrame([metrics]).to_csv(metrics_dir / "classifier_metrics.csv", index=False)
	pd.DataFrame(confusion_matrix(y_test, predictions), index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"]).to_csv(metrics_dir / "confusion_matrix.csv")

	fpr, tpr, _ = roc_curve(y_test, probabilities)
	plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.savefig(figures_dir / "roc_curve.png", dpi=140, bbox_inches="tight"); plt.close()
	precision, recall, _ = precision_recall_curve(y_test, probabilities)
	plt.figure(); plt.plot(recall, precision); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.savefig(figures_dir / "precision_recall_curve.png", dpi=140, bbox_inches="tight"); plt.close()
	frac_pos, mean_pred = calibration_curve(y_test, probabilities, n_bins=10)
	plt.figure(); plt.plot(mean_pred, frac_pos, marker="o"); plt.plot([0, 1], [0, 1], "--"); plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction Positive"); plt.title("Calibration Curve"); plt.savefig(figures_dir / "calibration_curve.png", dpi=140, bbox_inches="tight"); plt.close()
	plt.figure(); plt.hist(probabilities, bins=20); plt.xlabel("Failure Probability"); plt.title("Failure Probability Distribution"); plt.savefig(figures_dir / "failure_probability_distribution.png", dpi=140, bbox_inches="tight"); plt.close()
	for column, title, filename in [("risk_score", "Risk Score Distribution", "risk_score_distribution.png")]:
		plt.figure(); plt.hist(results[column], bins=20); plt.xlabel(column); plt.title(title); plt.savefig(figures_dir / filename, dpi=140, bbox_inches="tight"); plt.close()
	results["action"].value_counts().reindex(["Normal", "Scale Resources", "Migrate VM", "Restart Task"], fill_value=0).plot.bar(); plt.title("Action Distribution"); plt.savefig(figures_dir / "action_distribution.png", dpi=140, bbox_inches="tight"); plt.close()
	results["risk_band"] = results["risk_score"].map(
		lambda value: "Normal" if value < 0.30 else "Scale Resources" if value < 0.50 else "Migrate VM" if value < 0.65 else "Restart Task"
	)
	results["risk_band"].value_counts().reindex(["Normal", "Scale Resources", "Migrate VM", "Restart Task"], fill_value=0).plot.bar(); plt.title("Risk-band Distribution"); plt.savefig(figures_dir / "risk_band_distribution.png", dpi=140, bbox_inches="tight"); plt.close()
	cm = confusion_matrix(y_test, predictions)
	plt.figure(); plt.imshow(cm, cmap="Blues"); plt.colorbar(); plt.xticks([0, 1], ["0", "1"]); plt.yticks([0, 1], ["0", "1"]); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix"); plt.savefig(figures_dir / "confusion_matrix.png", dpi=140, bbox_inches="tight"); plt.close()
	importance = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values().tail(15)
	importance.plot.barh(); plt.title("Random Forest Feature Importance"); plt.savefig(figures_dir / "feature_importance.png", dpi=140, bbox_inches="tight"); plt.close()
	return metrics


def plot_latency_outputs(timings, output_dir="results/latency"):
	"""Create latency distribution and component-breakdown figures."""
	from pathlib import Path
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	path = Path(output_dir)
	path.mkdir(parents=True, exist_ok=True)
	plt.figure(); plt.hist(timings["total_ms"], bins=min(20, max(3, len(timings)))); plt.xlabel("Total latency (ms)"); plt.title("Latency Distribution"); plt.savefig(path / "latency_distribution.png", dpi=140, bbox_inches="tight"); plt.close()
	components = ["rf_ms", "isolation_forest_ms", "calibration_ms", "risk_ms", "decision_ms", "rca_ms", "forecast_ms"]
	plt.figure(); timings[components].mean().plot.bar(); plt.ylabel("Mean latency (ms)"); plt.title("Latency Component Breakdown"); plt.savefig(path / "latency_component_breakdown.png", dpi=140, bbox_inches="tight"); plt.close()
