import json
import os

import pandas as pd

from src.preprocessing import preprocess_data
from src.train_model import train_model
from src.anomaly_detection import train_anomaly_model
from src.predict import generate_results
from src.diagnosis import explain_prediction
from src.evaluate import benchmark_components, latency_summary_table, paper_evaluation, plot_latency_outputs
from src.forecasting import evaluate_temporal_forecast, forecast_load, load_ordered_load_series, temporal_split
from config import DATA_PATH
DATA_PATH = "data/raw/borg_traces_data.csv"

def main():

    print("Preprocessing data...")
    df = preprocess_data(DATA_PATH)

    print("Training model...")
    model, X_train, X_test, y_test = train_model(df)

    print("Training anomaly model...")
    iso = train_anomaly_model(X_train)

    print("Generating results...")
    results = generate_results(model, iso, X_test)

    results.to_csv("data/processed/results.csv", index=False)
    metrics = paper_evaluation(model, X_test, y_test, results, output_dir="results")
    benchmark_features = X_test.head(min(1000, len(X_test)))
    latency = benchmark_components(model, iso, benchmark_features, repetitions=5)
    full_features = X_test.head(min(100, len(X_test)))
    full_latency = benchmark_components(
        model,
        iso,
        full_features,
        repetitions=1,
        explanation_fn=lambda row: explain_prediction(
            model, row.iloc[0].to_numpy(), anomaly_model=iso, feature_names=row.columns
        ),
    )
    latency["mode"] = "prediction_plus_anomaly_risk"
    full_latency["mode"] = "full_pipeline_with_explanation"
    latency = pd.concat([latency, full_latency], ignore_index=True)
    latency_dir = "results/latency"
    os.makedirs(latency_dir, exist_ok=True)
    latency.to_csv(f"{latency_dir}/latency_benchmark.csv", index=False)
    latency_summary_table(latency.drop(columns=["run_id", "mode"])).to_csv(
        f"{latency_dir}/latency_summary.csv", index=False
    )
    plot_latency_outputs(latency[latency["mode"] == "prediction_plus_anomaly_risk"])
    temporal = evaluate_temporal_forecast(load_ordered_load_series(DATA_PATH))
    os.makedirs("results/forecasting", exist_ok=True)
    with open("results/forecasting/forecast_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(temporal, handle, indent=2)
    ordered = load_ordered_load_series(DATA_PATH)
    train, test = temporal_split(ordered)
    forecast_figures = "results/forecasting"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for feature, label in (("cpu_load", "CPU"), ("memory_load", "Memory")):
        predicted = forecast_load(train[feature].to_numpy(), len(test))
        plt.figure(); plt.plot(test["start_time"], test[feature], label="Actual"); plt.plot(test["start_time"], predicted, label="Linear forecast"); plt.legend(); plt.xlabel("start_time"); plt.ylabel(label); plt.title(f"Actual vs predicted {label}"); plt.savefig(f"{forecast_figures}/actual_vs_predicted_{feature}.png", dpi=140, bbox_inches="tight"); plt.close()
    with open("results/metrics/classifier_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print("Results saved!")
    print("Classifier metrics:", metrics)
    print(results.head())

if __name__ == "__main__":
    main()