"""
evaluate.py
Load the saved best model and print a full evaluation report.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_loader import load_and_preprocess

MODELS_DIR = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def main():
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Run train.py first to generate the checkpoint.")

    with open(model_path, "rb") as f:
        checkpoint = pickle.load(f)

    model = checkpoint["model"]
    name = checkpoint["name"]
    features = checkpoint["features"]

    print(f"Loaded model: {name}")

    X, y = load_and_preprocess()
    # Keep only the features the model was trained on (same split)
    X = X[features]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Load stored metrics
    metrics_path = REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            all_metrics = json.load(f)
        print("\n── All Models Summary ──")
        df = pd.DataFrame(all_metrics).T.sort_values("roc_auc", ascending=False)
        print(df.to_string())


if __name__ == "__main__":
    main()
