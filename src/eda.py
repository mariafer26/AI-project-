"""
eda.py
Exploratory Data Analysis for Telco Customer Churn.
Generates plots saved to reports/figures/.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from data_loader import load_and_preprocess, DATA_PATH

FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")


def run_eda():
    # Load raw (before encoding) for nicer plots
    import numpy as np
    df_raw = pd.read_csv(DATA_PATH)
    df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")
    df_raw["Churn_bin"] = (df_raw["Churn"] == "Yes").astype(int)

    # ── Figure 1: Churn distribution ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df_raw["Churn"].value_counts()
    bars = ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"], width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val}\n({val/len(df_raw):.1%})", ha="center", va="bottom", fontsize=10)
    ax.set_title("Churn Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Customers")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_distribution.png", dpi=150)
    plt.close()
    print("Saved: churn_distribution.png")

    # ── Figure 2: Tenure by Churn ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, grp in df_raw.groupby("Churn")["tenure"]:
        ax.hist(grp, bins=24, alpha=0.7, label=label)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Count")
    ax.set_title("Tenure Distribution by Churn", fontsize=13, fontweight="bold")
    ax.legend(title="Churn")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tenure_by_churn.png", dpi=150)
    plt.close()
    print("Saved: tenure_by_churn.png")

    # ── Figure 3: MonthlyCharges by Churn ────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_raw, x="Churn", y="MonthlyCharges", ax=ax,
                hue="Churn", palette={"No": "#4C72B0", "Yes": "#DD8452"}, legend=False)
    ax.set_title("Monthly Charges by Churn", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_charges_by_churn.png", dpi=150)
    plt.close()
    print("Saved: monthly_charges_by_churn.png")

    # ── Figure 4: Churn rate by Contract type ────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ct = df_raw.groupby("Contract")["Churn_bin"].mean().sort_values(ascending=False)
    ax.bar(ct.index, ct.values * 100, color="#4C72B0", width=0.5)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Churn Rate by Contract Type", fontsize=13, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_by_contract.png", dpi=150)
    plt.close()
    print("Saved: churn_by_contract.png")

    # ── Summary stats ────────────────────────────────────────────────────
    print("\n── Dataset Summary ──")
    print(f"Total customers : {len(df_raw):,}")
    print(f"Features        : {df_raw.shape[1] - 2}")
    print(f"Churn rate      : {df_raw['Churn_bin'].mean():.2%}")
    print(f"Missing values  : {df_raw.isnull().sum().sum()}")


if __name__ == "__main__":
    run_eda()
