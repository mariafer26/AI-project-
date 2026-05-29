"""
agent.py
AI Agent: takes a customer profile + model prediction and generates
a natural-language explanation of churn risk and retention suggestions.

Usage:
    python agent.py                         # random customer from test set
    python agent.py --customer-id <index>   # specific row index
    python agent.py --interactive           # enter values manually
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import load_and_preprocess, DATA_PATH

MODELS_DIR = Path(__file__).parent.parent / "models"

# ── Rule-based explanation engine ─────────────────────────────────────────
RISK_FACTORS = {
    "Contract_Month-to-month": (
        "month-to-month contract",
        "Offer a discounted 1- or 2-year contract upgrade."
    ),
    "tenure": (
        "short customer tenure",
        "Engage with a loyalty program or early-bird rewards."
    ),
    "InternetService_Fiber optic": (
        "fiber-optic internet subscription (high churn segment)",
        "Proactively address any service quality concerns and offer service credits."
    ),
    "MonthlyCharges": (
        "high monthly charges",
        "Review the customer's plan and offer a tailored bundle at a lower price."
    ),
    "OnlineSecurity_No": (
        "no online security add-on",
        "Offer a free trial of Online Security to increase stickiness."
    ),
    "PaymentMethod_Electronic check": (
        "electronic check payment (correlated with higher churn)",
        "Encourage switch to automatic payment with a small bill credit."
    ),
}

RETENTION_ACTIONS = [
    "Schedule a proactive customer satisfaction call within 7 days.",
    "Send a personalized email highlighting loyalty benefits.",
    "Provide a temporary service upgrade at no extra cost.",
    "Assign a dedicated support representative.",
]


def explain(customer: pd.Series, churn_prob: float, feature_importances: dict) -> str:
    """Generate a plain-language churn explanation for a single customer."""
    risk_level = (
        "HIGH" if churn_prob >= 0.65 else
        "MEDIUM" if churn_prob >= 0.40 else
        "LOW"
    )

    lines = [
        "=" * 60,
        f"  CHURN RISK ASSESSMENT",
        "=" * 60,
        f"  Churn Probability : {churn_prob:.1%}",
        f"  Risk Level        : {risk_level}",
        "=" * 60,
        "",
        "📊 KEY RISK FACTORS:",
    ]

    # Match features in customer profile
    found_factors = []
    for feat, (description, action) in RISK_FACTORS.items():
        if feat == "tenure" and customer.get("tenure", 999) < 12:
            found_factors.append((description, action, feature_importances.get(feat, 0)))
        elif feat == "MonthlyCharges" and customer.get("MonthlyCharges", 0) > 70:
            found_factors.append((description, action, feature_importances.get(feat, 0)))
        elif feat in customer.index and customer[feat] == 1:
            found_factors.append((description, action, feature_importances.get(feat, 0)))

    if not found_factors:
        lines.append("  • No major individual risk factors identified.")
    else:
        for desc, _, _ in sorted(found_factors, key=lambda x: -x[2])[:3]:
            lines.append(f"  • Customer has {desc}.")

    lines += ["", "💡 SUGGESTED RETENTION ACTIONS:"]
    actions = [a for _, a, _ in sorted(found_factors, key=lambda x: -x[2])[:3]]
    if not actions:
        actions = random.sample(RETENTION_ACTIONS, 2)
    for i, action in enumerate(actions[:3], 1):
        lines.append(f"  {i}. {action}")

    if risk_level == "HIGH":
        lines += [
            "",
            "⚠️  PRIORITY: This customer is at high risk of churning within 30 days.",
            "   Escalate to the retention team immediately.",
        ]

    lines.append("=" * 60)
    return "\n".join(lines)


def get_feature_importances(model) -> dict:
    """Extract feature importances from the underlying estimator."""
    estimator = model
    if hasattr(model, "named_steps"):  # Pipeline
        estimator = model.named_steps.get("clf", model)
    if hasattr(estimator, "feature_importances_"):
        return dict(enumerate(estimator.feature_importances_))
    if hasattr(estimator, "coef_"):
        return {i: abs(c) for i, c in enumerate(estimator.coef_[0])}
    return {}


def run_agent(customer_idx: int | None = None):
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Run train.py first to generate the checkpoint.")

    with open(model_path, "rb") as f:
        checkpoint = pickle.load(f)

    model = checkpoint["model"]
    features = checkpoint["features"]

    X, y = load_and_preprocess()
    X = X[features]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    if customer_idx is None:
        customer_idx = random.randint(0, len(X_test) - 1)

    customer = X_test.iloc[customer_idx]
    true_label = y_test.iloc[customer_idx]
    churn_prob = model.predict_proba(customer.to_frame().T)[0, 1]

    # Named importances
    raw_importances = get_feature_importances(model)
    named_importances = {features[i]: v for i, v in raw_importances.items() if i < len(features)}

    print(f"\nCustomer index  : {customer_idx}")
    print(f"Actual churn    : {'Yes' if true_label == 1 else 'No'}")
    print()
    print(explain(customer, churn_prob, named_importances))


def main():
    parser = argparse.ArgumentParser(description="Churn AI Agent")
    parser.add_argument("--customer-id", type=int, default=None)
    args = parser.parse_args()
    run_agent(args.customer_id)


if __name__ == "__main__":
    main()
