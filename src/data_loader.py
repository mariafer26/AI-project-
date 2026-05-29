"""
data_loader.py
Downloads and prepares the Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import io

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_PATH = DATA_DIR / "telco_churn.csv"

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)


def download_dataset():
    """Download dataset from IBM GitHub if not present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        print(f"Dataset already at {DATA_PATH}")
        return
    print("Downloading Telco Customer Churn dataset...")
    urllib.request.urlretrieve(TELCO_URL, DATA_PATH)
    print(f"Saved to {DATA_PATH}")


def load_and_preprocess() -> tuple[pd.DataFrame, pd.Series]:
    """Load, clean and encode the dataset. Returns X, y."""
    if not DATA_PATH.exists():
        download_dataset()

    df = pd.read_csv(DATA_PATH)

    # Fix TotalCharges (has spaces for new customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop customerID
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Target
    y = (df["Churn"].str.strip() == "Yes").astype(int)
    df.drop(columns=["Churn"], inplace=True)

    # Encode binary columns
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in df.select_dtypes(include="str").columns:
        if df[col].nunique() == 2:
            df[col] = df[col].map(binary_map).fillna(df[col])

    # One-hot encode remaining categoricals
    df = pd.get_dummies(df, drop_first=True)

    return df.astype(float), y


if __name__ == "__main__":
    X, y = load_and_preprocess()
    print(f"Shape: {X.shape}, Churn rate: {y.mean():.2%}")
