import pandas as pd


UI_FEATURE_RANGES = {
    "priority": (0, 500),
    "cpu_mean": (0.0, 1.0),
    "assigned_memory": (0.0, 1.0),
    "cycles_per_instruction": (0.0, 2.0),
    "memory_accesses_per_instruction": (0.0, 1.0),
    "sample_rate": (0.0, 1.0),
}


def build_single_prediction_features(values, model_features, feature_medians):
    """Build the legacy synthetic row used only by boundary-analysis regression tests.

    The Streamlit Single Prediction UI uses ``src.profile_prediction`` instead.
    This function remains unchanged so the historical UI-reachability boundary
    analysis continues to document the former synthetic input space.
    """
    cpu_mean = values["cpu_mean"]
    memory = values["assigned_memory"]
    input_data = pd.DataFrame([{
        "priority": values["priority"],
        "cpu_mean": cpu_mean,
        "assigned_memory": memory,
        "cycles_per_instruction": values["cycles_per_instruction"],
        "memory_accesses_per_instruction": values["memory_accesses_per_instruction"],
        "sample_rate": values["sample_rate"],
        "avg_cpu": cpu_mean,
        "max_cpu": cpu_mean,
        "sample_cpu": cpu_mean,
        "tail_cpu_mean": cpu_mean,
        "req_cpu": cpu_mean,
        "avg_memory": memory,
        "max_memory": memory,
        "req_memory": memory,
        "page_cache_memory": memory * 0.1,
    }])
    for column in model_features:
        if column not in input_data.columns:
            input_data[column] = feature_medians[column]
    return input_data[list(model_features)]
