"""
Exploratory Data Analysis (EDA) Module
Generates comprehensive distribution plots, correlation matrices,
box plots, and temporal behavioral sequence visualizations.
"""

import json
from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import (
    DATASET_PATH,
    REPORTS_DIR,
    EDA_DIR,
    FEATURES,
    SEQ_LEN,
)

# Set high-quality plotting aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Generate and save class balance chart."""
    seq_classes = df.groupby("sequence_id")["label"].first().value_counts()
    labels = ["BENIGN (0)", "RANSOMWARE (1)"]
    counts = [seq_classes.get(0, 0), seq_classes.get(1, 0)]
    colors = ["#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="#333333", alpha=0.9)
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval / 2,
            f"{yval:,} seqs\n({yval/sum(counts)*100:.1f}%)",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=12,
        )

    ax.set_title("Sequence Class Distribution (Ground Truth)", fontsize=14, pad=15, fontweight="bold")
    ax.set_ylabel("Number of Sequences", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()

    out_path = EDA_DIR / "class_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Generate full feature correlation heatmap."""
    corr = df[FEATURES].corr()

    fig, ax = plt.subplots(figsize=(16, 13), dpi=300)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-0.5,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation Coefficient"},
        annot=False,
        ax=ax,
    )
    ax.set_title("Behavioral Telemetry Feature Correlation Matrix", fontsize=16, pad=15, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    # Save to both standard report path and eda folder as per PRD
    out_main = REPORTS_DIR / "correlation_matrix.png"
    out_eda = EDA_DIR / "correlation_matrix.png"
    plt.savefig(out_main, dpi=300)
    plt.savefig(out_eda, dpi=300)
    plt.close()
    print(f"[+] Saved: {out_main}")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Plot histograms for selected critical ransomware indicators."""
    key_features = [
        "file_write_rate",
        "file_rename_rate",
        "file_delete_rate",
        "entropy_delta",
        "extension_change_rate",
        "encrypted_file_ratio",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    axes = axes.flatten()

    for i, feat in enumerate(key_features):
        ax = axes[i]
        sns.kdeplot(
            data=df,
            x=feat,
            hue="label",
            fill=True,
            common_norm=False,
            palette={0: "#2ecc71", 1: "#e74c3c"},
            ax=ax,
            alpha=0.4,
            linewidth=1.5,
        )
        ax.set_title(f"Distribution: {feat}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        # Custom legend labels
        ax.legend(["Ransomware", "Benign"], loc="upper right")

    plt.suptitle("Feature Distributions: Benign vs Ransomware", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_path = EDA_DIR / "feature_distributions.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_feature_boxplots(df: pd.DataFrame) -> None:
    """Generate comparative boxplots for prominent telemetry metrics."""
    compare_features = [
        "file_write_rate",
        "entropy_delta",
        "encrypted_file_ratio",
        "cpu_percent",
    ]

    plot_df = df.copy()
    plot_df["Class"] = plot_df["label"].map({0: "Benign", 1: "Ransomware"})

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), dpi=300)

    for i, feat in enumerate(compare_features):
        ax = axes[i]
        sns.boxplot(
            data=plot_df,
            x="Class",
            y=feat,
            hue="Class",
            palette={"Benign": "#2ecc71", "Ransomware": "#e74c3c"},
            ax=ax,
            showfliers=False,
            width=0.4,
            legend=False,
        )
        ax.set_title(f"{feat}", fontsize=12, fontweight="bold")
        ax.set_xlabel("")

    plt.suptitle("Behavioral Feature Range Comparison (Outliers Hidden)", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_path = EDA_DIR / "feature_boxplots.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_temporal_progression(df: pd.DataFrame) -> None:
    """
    Plot mean trajectory and standard deviations of file_write_rate,
    entropy_delta, and encrypted_file_ratio across the 20 time steps.
    """
    # Group by label and time_step
    prog = df.groupby(["label", "time_step"]).agg({
        "file_write_rate": ["mean", "std"],
        "entropy_delta": ["mean", "std"],
        "encrypted_file_ratio": ["mean", "std"],
    })

    timesteps = np.arange(SEQ_LEN)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=300)

    metrics = [
        ("file_write_rate", "File Write Rate (ops/sec)", axes[0]),
        ("entropy_delta", "File Entropy Delta", axes[1]),
        ("encrypted_file_ratio", "Encrypted File Ratio", axes[2]),
    ]

    for col, title, ax in metrics:
        # Benign (label 0)
        b_mean = prog.loc[0][col]["mean"].values
        b_std = prog.loc[0][col]["std"].values
        ax.plot(timesteps, b_mean, color="#2ecc71", label="Benign (Mean)", linewidth=2.5)
        ax.fill_between(timesteps, b_mean - b_std * 0.5, b_mean + b_std * 0.5, color="#2ecc71", alpha=0.15)

        # Ransomware (label 1)
        r_mean = prog.loc[1][col]["mean"].values
        r_std = prog.loc[1][col]["std"].values
        ax.plot(timesteps, r_mean, color="#e74c3c", label="Ransomware (Mean)", linewidth=2.5, linestyle="--")
        ax.fill_between(timesteps, r_mean - r_std * 0.5, r_mean + r_std * 0.5, color="#e74c3c", alpha=0.15)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time Step (Window 0 to 19)")
        ax.set_ylabel("Value")
        ax.set_xticks(range(0, SEQ_LEN, 2))
        ax.legend(loc="upper left")

    plt.suptitle("Temporal Behavioral Progression Over 20 Time Windows", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_path = EDA_DIR / "sequence_temporal_patterns.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {out_path}")


def run_eda(csv_path: Optional[Path] = None) -> None:
    """Execute complete EDA pipeline and output artifacts."""
    path = csv_path or DATASET_PATH
    print(f"[*] Executing Exploratory Data Analysis on: {path}...")
    df = pd.read_csv(path)

    plot_class_distribution(df)
    plot_correlation_matrix(df)
    plot_feature_distributions(df)
    plot_feature_boxplots(df)
    plot_temporal_progression(df)

    # Save summary stats JSON
    summary_stats = {
        "total_rows": int(len(df)),
        "total_sequences": int(df["sequence_id"].nunique()),
        "features": FEATURES,
        "class_breakdown": df.groupby("sequence_id")["label"].first().value_counts().to_dict(),
    }
    with open(EDA_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=4)

    print("[+] EDA pipeline completed successfully.")


if __name__ == "__main__":
    run_eda()
