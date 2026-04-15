"""
Generate charts that explain churn patterns and model performance.

All figures are saved to the visuals/ folder so the notebook
and README can reference them.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

VISUALS_DIR = Path(__file__).resolve().parent.parent / "visuals"
VISUALS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def plot_churn_distribution(df: pd.DataFrame):
    """Bar chart showing how many customers stayed vs. left."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Churn"].value_counts()
    labels = ["Stayed", "Churned"]
    colors = ["#2ecc71", "#e74c3c"]
    ax.bar(labels, counts.values, color=colors, edgecolor="black", linewidth=0.5)

    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontweight="bold")

    ax.set_title("Customer Churn Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Customers")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "churn_distribution.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model: DecisionTreeClassifier, feature_names: list):
    """Horizontal bar chart of the top features driving churn predictions."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="#3498db",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_title("Top 10 Features Driving Churn", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def plot_confusion(model, X_test, y_test):
    """Confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_monthly_charges_by_churn(df: pd.DataFrame):
    """Box plot comparing monthly charges for customers who stayed vs. left."""
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_df = df.copy()
    plot_df["Churn"] = plot_df["Churn"].map({0: "Stayed", 1: "Churned"})
    sns.boxplot(data=plot_df, x="Churn", y="MonthlyCharges", ax=ax,
                palette={"Stayed": "#2ecc71", "Churned": "#e74c3c"})
    ax.set_title("Monthly Charges: Stayed vs Churned", fontsize=14, fontweight="bold")
    ax.set_ylabel("Monthly Charges ($)")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "monthly_charges_boxplot.png", dpi=150)
    plt.close(fig)


def plot_tenure_by_churn(df: pd.DataFrame):
    """Histogram of customer tenure split by churn status."""
    fig, ax = plt.subplots(figsize=(7, 4))
    stayed = df[df["Churn"] == 0]["tenure"]
    churned = df[df["Churn"] == 1]["tenure"]
    ax.hist(stayed, bins=30, alpha=0.6, label="Stayed", color="#2ecc71", edgecolor="black")
    ax.hist(churned, bins=30, alpha=0.6, label="Churned", color="#e74c3c", edgecolor="black")
    ax.set_title("Customer Tenure Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Number of Customers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "tenure_distribution.png", dpi=150)
    plt.close(fig)


def plot_decision_tree(model: DecisionTreeClassifier, feature_names: list):
    """Compact view of the trained decision tree (top 3 levels)."""
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        model,
        max_depth=3,
        feature_names=feature_names,
        class_names=["Stayed", "Churned"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    ax.set_title("Decision Tree (Top 3 Levels)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "decision_tree.png", dpi=150)
    plt.close(fig)


def generate_all_visuals(model, X_test, y_test, df, feature_names):
    """One-call convenience function to produce every chart."""
    plot_churn_distribution(df)
    plot_feature_importance(model, feature_names)
    plot_confusion(model, X_test, y_test)
    plot_monthly_charges_by_churn(df)
    plot_tenure_by_churn(df)
    plot_decision_tree(model, feature_names)
    print(f"All visuals saved to {VISUALS_DIR}/")
