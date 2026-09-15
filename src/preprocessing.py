import pandas as pd
import numpy as np
import ast
import re

def extract_cpu(x):
    try:
        data = ast.literal_eval(str(x))
        return data.get("cpus", 0)
    except:
        return 0

def extract_memory(x):
    try:
        data = ast.literal_eval(str(x))
        return data.get("memory", 0)
    except:
        return 0

def extract_mean(x):
    try:
        nums = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", str(x))
        nums = [float(i) for i in nums]
        return np.mean(nums) if len(nums) > 0 else 0
    except:
        return 0

def preprocess_data(path, nrows=100000):

    df = pd.read_csv(path, nrows=nrows)

    # Feature engineering
    df["req_cpu"] = df["resource_request"].apply(extract_cpu)
    df["req_memory"] = df["resource_request"].apply(extract_memory)

    df["avg_cpu"] = df["average_usage"].apply(extract_cpu)
    df["avg_memory"] = df["average_usage"].apply(extract_memory)

    df["max_cpu"] = df["maximum_usage"].apply(extract_cpu)
    df["max_memory"] = df["maximum_usage"].apply(extract_memory)

    df["sample_cpu"] = df["random_sample_usage"].apply(extract_cpu)
    df["sample_memory"] = df["random_sample_usage"].apply(extract_memory)

    df["cpu_mean"] = df["cpu_usage_distribution"].apply(extract_mean)
    df["tail_cpu_mean"] = df["tail_cpu_usage_distribution"].apply(extract_mean)

    # Drop leakage + raw columns
    drop_cols = [
        "resource_request","average_usage","maximum_usage",
        "random_sample_usage","cpu_usage_distribution",
        "tail_cpu_usage_distribution","event","time",
        "start_time","end_time","cluster",
        "instance_events_type","collections_events_type"
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    df = df.fillna(0)

    df = pd.get_dummies(
        df,
        columns=["scheduling_class","collection_type","scheduler"],
        drop_first=True
    )

    non_numeric_cols = df.select_dtypes(include=["object"]).columns

    if len(non_numeric_cols) > 0:
        print("Dropping non-numeric columns:", non_numeric_cols)
        df = df.drop(columns=non_numeric_cols)
    # Keep numeric columns in their native dtype (float64) so continuous
    # features such as assigned_memory and cycles_per_instruction retain
    # their variance.  The previous .astype(int) truncated them all to 0.
    return df