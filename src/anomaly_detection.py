import os
import joblib
from sklearn.ensemble import IsolationForest
from config import ANOMALY_PATH

def train_anomaly_model(X_train):

    if os.path.exists("models/anomaly_model.pkl"):
        print("Loading anomaly model...")
        return joblib.load("models/anomaly_model.pkl")

    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    iso.fit(X_train)

    joblib.dump(iso, "models/anomaly_model.pkl")

    return iso