import os
import joblib
from config import MODEL_PATH
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(df):

    if os.path.exists("models/failure_model.pkl"):
        print("Loading saved model...")
        model = joblib.load("models/failure_model.pkl")

        X = df.drop("failed", axis=1)
        y = df["failed"]
        X_train, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return model, X_train, X_test, y_test

    print("Training new model...")

    X = df.drop("failed", axis=1)
    y = df["failed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/failure_model.pkl")

    return model, X_train, X_test, y_test