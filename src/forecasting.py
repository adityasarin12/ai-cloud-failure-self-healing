import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.preprocessing import extract_cpu, extract_memory


def forecast_load(values, horizon=1):
    """Forecast numeric load values; requires ordered telemetry, not failure labels."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("at least three ordered load observations are required")
    x = np.arange(len(values)).reshape(-1, 1)
    model = LinearRegression().fit(x, values)
    future = np.arange(len(values), len(values) + horizon).reshape(-1, 1)
    return np.maximum(model.predict(future), 0.0)


def load_ordered_load_series(path, nrows=100000):
    """Extract real usage observations ordered by Borg start time."""
    columns = ["start_time", "end_time", "average_usage", "maximum_usage"]
    raw = pd.read_csv(path, nrows=nrows, usecols=lambda column: column in columns)
    raw["cpu_load"] = raw["average_usage"].map(extract_cpu).astype(float)
    raw["memory_load"] = raw["average_usage"].map(extract_memory).astype(float)
    ordered = raw.sort_values(["start_time", "end_time"], kind="mergesort")
    return (
        ordered.groupby("start_time", as_index=False)[["cpu_load", "memory_load"]]
        .mean()
        .sort_values("start_time")
        .reset_index(drop=True)
    )


def temporal_split(series, holdout_fraction=0.2):
    """Split an ordered series without using observations from the future."""
    if len(series) < 10:
        raise ValueError("at least ten ordered observations are required")
    split_at = max(1, int(len(series) * (1.0 - holdout_fraction)))
    return series.iloc[:split_at].copy(), series.iloc[split_at:].copy()


def _forecast_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = actual != 0
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else None,
    }


def evaluate_temporal_forecast(series, holdout_fraction=0.2):
    """Evaluate linear forecasting against persistence on a chronological holdout."""
    train, test = temporal_split(series, holdout_fraction)
    result = {"train_rows": len(train), "test_rows": len(test), "split_is_chronological": bool(train.start_time.max() <= test.start_time.min()), "metrics": {}}
    for feature in ("cpu_load", "memory_load"):
        linear = forecast_load(train[feature].to_numpy(), horizon=len(test))
        persistence = np.repeat(train[feature].iloc[-1], len(test))
        result["metrics"][feature] = {
            "linear": _forecast_metrics(test[feature], linear),
            "persistence": _forecast_metrics(test[feature], persistence),
        }
    return result


def forecast_cpu_memory(series, horizon=3, window=50):
    """Forecast CPU and memory independently from the latest real observations."""
    if len(series) < 3:
        raise ValueError("at least three ordered observations are required")
    recent = series.tail(window)
    return {
        "current_cpu": float(recent["cpu_load"].iloc[-1]),
        "current_memory": float(recent["memory_load"].iloc[-1]),
        "forecast_cpu": forecast_load(recent["cpu_load"].to_numpy(), horizon).tolist(),
        "forecast_memory": forecast_load(recent["memory_load"].to_numpy(), horizon).tolist(),
    }


def forecast_status(forecast):
    """Classify the independent forecast without affecting failure-risk policy."""
    cpu = np.asarray(forecast["forecast_cpu"], dtype=float)
    memory = np.asarray(forecast["forecast_memory"], dtype=float)
    current = max(forecast["current_cpu"], forecast["current_memory"], 1e-12)
    projected = max(cpu.max(initial=0.0), memory.max(initial=0.0))
    increasing = (cpu[-1] > forecast["current_cpu"] * 1.05) or (memory[-1] > forecast["current_memory"] * 1.05)
    if projected >= 0.9:
        return "Critical load expected"
    if projected >= 0.7:
        return "High load expected"
    if increasing and projected > current:
        return "Increasing"
    return "Stable"


def proactive_recommendation(current_action, forecast):
    """Return a forecast-only recommendation, distinct from current risk action."""
    status = forecast_status(forecast)
    if status in {"High load expected", "Critical load expected"}:
        return "Scale Resources proactively"
    if status == "Increasing":
        return "Prepare for Increased Load"
    return "No proactive action indicated"