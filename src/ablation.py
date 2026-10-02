"""
Ablation Study Module
Systematically assesses the contribution of each architectural component:
  A: 1D-CNN Only
  B: BiLSTM Only
  C: 1D-CNN + BiLSTM (No Attention)
  D: 1D-CNN + BiLSTM + Attention (Proposed Hybrid Model)
Generates reports/ablation_results.csv and reports/ablation_comparison.png.
"""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    SEED,
    BATCH_SIZE,
    DEVICE,
    ABLATION_RESULTS_PATH,
    ABLATION_COMPARISON_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_CHECKPOINT_PATH,
    SCALER_PATH,
    N_FEATURES,
    CNN_FILTERS,
    LSTM_HIDDEN_SIZE,
)
from src.preprocessing import (
    load_raw_dataset,
    split_sequences,
    load_scaler,
    get_dataloaders,
)
from src.model import (
    CNNOnlyModel,
    BiLSTMOnlyModel,
    CNN_BiLSTMModel,
    RansomwareHybridModel,
)
from src.evaluate import load_trained_model, get_predictions, compute_metrics
from src.baselines import train_eval_torch_model


def plot_ablation_comparison(df: pd.DataFrame) -> None:
    """Generate and save grouped bar chart for ablation metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    df_melted = pd.melt(df, id_vars=["Architecture"], value_vars=metrics, var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    palette = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"]

    sns.barplot(
        data=df_melted,
        x="Architecture",
        y="Score",
        hue="Metric",
        palette=palette,
        ax=ax,
        edgecolor="#333333",
        alpha=0.9,
    )

    ax.set_title("Architectural Ablation Study Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Architecture Configuration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax.set_ylim(0.70, 1.02)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(ABLATION_COMPARISON_PATH, dpi=300)
    plt.close()
    print(f"[+] Ablation comparison plot saved to: {ABLATION_COMPARISON_PATH}")


def run_ablation_study() -> pd.DataFrame:
    """Run ablation experiment across Configurations A, B, C, and D."""
    print("=" * 60)
    print("ARCHITECTURAL ABLATION STUDY")
    print("=" * 60)

    # If model_comparison.csv already exists, we can pull matching architectures directly
    # or evaluate freshly to ensure consistency
    df = load_raw_dataset()
    train_df, val_df, test_df = split_sequences(df, seed=SEED)
    scaler = load_scaler(SCALER_PATH)

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        train_df, val_df, test_df, scaler, batch_size=BATCH_SIZE
    )

    ablation_records: List[Dict[str, Any]] = []

    # Config A: CNN Only
    print("[*] Evaluating Ablation A: CNN Only...")
    cnn = CNNOnlyModel(in_features=N_FEATURES, cnn_filters=CNN_FILTERS)
    res_a = train_eval_torch_model(cnn, train_loader, val_loader, test_loader, epochs=12)
    res_a["Architecture"] = "A: CNN Only"
    ablation_records.append(res_a)

    # Config B: BiLSTM Only
    print("[*] Evaluating Ablation B: BiLSTM Only...")
    lstm = BiLSTMOnlyModel(in_features=N_FEATURES, lstm_hidden=LSTM_HIDDEN_SIZE)
    res_b = train_eval_torch_model(lstm, train_loader, val_loader, test_loader, epochs=12)
    res_b["Architecture"] = "B: BiLSTM Only"
    ablation_records.append(res_b)

    # Config C: CNN + BiLSTM
    print("[*] Evaluating Ablation C: CNN + BiLSTM (No Attention)...")
    cnn_lstm = CNN_BiLSTMModel(in_features=N_FEATURES, cnn_filters=CNN_FILTERS, lstm_hidden=LSTM_HIDDEN_SIZE)
    res_c = train_eval_torch_model(cnn_lstm, train_loader, val_loader, test_loader, epochs=12)
    res_c["Architecture"] = "C: CNN + BiLSTM"
    ablation_records.append(res_c)

    # Config D: CNN + BiLSTM + Attention (Proposed)
    print("[*] Evaluating Ablation D: CNN + BiLSTM + Attention (Proposed)...")
    try:
        model_d = load_trained_model(MODEL_CHECKPOINT_PATH, device=DEVICE)
        probs, targets, _ = get_predictions(model_d, test_loader, device=DEVICE)
        res_d_m = compute_metrics(targets, probs)
        res_d = {
            "Architecture": "D: CNN + BiLSTM + Attention",
            "Accuracy": res_d_m["accuracy"],
            "Precision": res_d_m["precision"],
            "Recall": res_d_m["recall"],
            "F1": res_d_m["f1"],
            "ROC-AUC": res_d_m["roc_auc"],
        }
    except Exception:
        model_d = RansomwareHybridModel().to(DEVICE)
        res_d = train_eval_torch_model(model_d, train_loader, val_loader, test_loader, epochs=12)
        res_d["Architecture"] = "D: CNN + BiLSTM + Attention"

    ablation_records.append(res_d)

    ablation_df = pd.DataFrame(ablation_records)
    ablation_df = ablation_df[["Architecture", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]

    for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        ablation_df[col] = ablation_df[col].round(4)

    ablation_df.to_csv(ABLATION_RESULTS_PATH, index=False)
    print(f"\n[+] Ablation results saved to: {ABLATION_RESULTS_PATH}")
    print(ablation_df.to_string(index=False))

    plot_ablation_comparison(ablation_df)

    return ablation_df


if __name__ == "__main__":
    run_ablation_study()
