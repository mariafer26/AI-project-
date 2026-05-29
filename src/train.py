"""
train.py
Trains Logistic Regression, Random Forest, and Gradient Boosting on the Telco Churn dataset.
Saves best model checkpoint to models/.
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)

from data_loader import load_and_preprocess

MODELS_DIR = Path(__file__).parent.parent / "models"
FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5

MODELS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=0.5)),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=4,
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=RANDOM_STATE,
    ),
}


def evaluate(name, model, X_test, y_test, results):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }
    results[name] = metrics
    print(f"\n── {name} ──")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    return y_prob


def plot_roc_curves(models_probs, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_prob in models_probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close()
    print("Saved: roc_curves.png")


def plot_confusion_matrix(model, name, X_test, y_test):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, ax=ax,
        display_labels=["No Churn", "Churn"],
        colorbar=False, cmap="Blues",
    )
    ax.set_title(f"Confusion Matrix – {name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(FIGURES_DIR / f"cm_{fname}.png", dpi=150)
    plt.close()
    print(f"Saved: cm_{fname}.png")


def main():
    print("Loading data...")
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results = {}
    models_probs = {}
    trained_models = {}

    for name, model in MODELS.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_prob = evaluate(name, model, X_test, y_test, results)
        models_probs[name] = y_prob
        trained_models[name] = model
        plot_confusion_matrix(model, name, X_test, y_test)

    # ROC curves
    plot_roc_curves(models_probs, y_test)

    # Save results
    results_path = Path(__file__).parent.parent / "reports" / "metrics.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {results_path}")

    # Save best model (by ROC-AUC)
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained_models[best_name]
    model_path = MODELS_DIR / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"name": best_name, "model": best_model, "features": list(X.columns)}, f)
    print(f"\nBest model: {best_name} (AUC={results[best_name]['roc_auc']:.4f})")
    print(f"Checkpoint saved to {model_path}")

    return results


if __name__ == "__main__":
    main()
