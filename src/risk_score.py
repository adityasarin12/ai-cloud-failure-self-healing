import os
import joblib
import numpy as np


class AnomalyCalibrator:
    """
    Empirical CDF / Quantile-based Anomaly Normalizer.
    Fitted strictly on training decision scores.
    Lower raw decision scores (anomalies) map monotonically to 1.0 (high anomaly risk).
    Higher raw decision scores (normal inliers) map monotonically to 0.0 (low anomaly risk).
    """

    def __init__(self, train_scores):
        self.reference_scores = np.sort(np.asarray(train_scores, dtype=float))
        self.n_samples = len(self.reference_scores)
        self.min_score = float(self.reference_scores[0]) if self.n_samples > 0 else -1.0
        self.max_score = float(self.reference_scores[-1]) if self.n_samples > 0 else 1.0

    def transform(self, scores):
        is_scalar = np.ndim(scores) == 0
        arr = np.asarray(scores, dtype=float)
        ranks = np.searchsorted(self.reference_scores, arr, side="right")
        percentile = ranks / max(1, self.n_samples)
        anomaly_signal = 1.0 - percentile
        clipped = np.clip(anomaly_signal, 0.0, 1.0)
        return float(clipped) if is_scalar else clipped


_CALIBRATOR = None


def get_anomaly_calibrator():
    global _CALIBRATOR
    if _CALIBRATOR is None:
        calibrator_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "anomaly_calibrator.pkl"
        )
        if os.path.exists(calibrator_path):
            _CALIBRATOR = joblib.load(calibrator_path)
    return _CALIBRATOR


def _normalize_anomaly_score(anomaly_score):
    """Convert Isolation Forest decision scores into a dataset-calibrated anomaly signal in [0, 1]."""
    calibrator = get_anomaly_calibrator()
    if calibrator is not None:
        return calibrator.transform(anomaly_score)

    # Fallback to empirical reference bounds if calibrator artifact is not yet saved
    score = np.asarray(anomaly_score, dtype=float)
    normalized = (0.17 - score) / (0.17 - (-0.20))
    clipped = np.clip(normalized, 0.0, 1.0)
    return float(clipped) if np.ndim(anomaly_score) == 0 else clipped


def calculate_risk(failure_prob, anomaly_score, failure_weight=0.7, anomaly_weight=0.3):
    """Combine failure probability and calibrated anomaly evidence into a [0, 1] score."""
    if not np.isclose(failure_weight + anomaly_weight, 1.0):
        raise ValueError("failure_weight and anomaly_weight must sum to 1")

    norm_anom = _normalize_anomaly_score(anomaly_score)
    risk = (
        failure_weight * np.asarray(failure_prob, dtype=float)
        + anomaly_weight * np.asarray(norm_anom, dtype=float)
    )
    risk = np.clip(risk, 0.0, 1.0)
    return float(risk) if np.ndim(risk) == 0 else risk


def self_heal(risk):

    if risk > 0.65:
        return "Restart Task"

    elif risk > 0.50:
        return "Migrate VM"

    elif risk > 0.30:
        return "Scale Resources"

    else:
        return "Normal"