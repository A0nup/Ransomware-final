"""
Evaluation Module
Computes standard cybersecurity & machine learning metrics on held-out test data.
Generates ROC curve, Precision-Recall curve, Confusion Matrix, and classification report.
"""

from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from src.config import (
    MODEL_CHECKPOINT_PATH,
    SCALER_PATH,
    CONFUSION_MATRIX_PATH,
    ROC_CURVE_PATH,
    PR_CURVE_PATH,
    CLASSIFICATION_REPORT_PATH,
    DEFAULT_THRESHOLD,
    DEVICE,
    SEED,
    BATCH_SIZE,
    N_FEATURES,
    CNN_FILTERS,
    LSTM_HIDDEN_SIZE,
    LSTM_LAYERS,
    DENSE_UNITS,
    DROPOUT_RATE,
)
from src.preprocessing import (
    load_raw_dataset,
    split_sequences,
    load_scaler,
    get_dataloaders,
)
from src.model import RansomwareHybridModel


def load_trained_model(checkpoint_path=MODEL_CHECKPOINT_PATH, device=DEVICE) -> RansomwareHybridModel:
    """Load model architecture and weights from serialized checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    params = checkpoint.get("architecture_params", {})

    model = RansomwareHybridModel(
        in_features=params.get("in_features", N_FEATURES),
        cnn_filters=params.get("cnn_filters", CNN_FILTERS),
        lstm_hidden=params.get("lstm_hidden", LSTM_HIDDEN_SIZE),
        lstm_layers=params.get("lstm_layers", LSTM_LAYERS),
        dense_units=params.get("dense_units", DENSE_UNITS),
        dropout=params.get("dropout", DROPOUT_RATE),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_predictions(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str = DEVICE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference on DataLoader to collect probabilities and ground truth."""
    model.eval()
    all_probs = []
    all_targets = []
    all_attns = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits, attn = model(X_batch)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

            all_probs.extend(probs)
            all_targets.extend(y_batch.numpy())
            if attn is not None:
                all_attns.extend(attn.cpu().numpy())

    return np.array(all_probs), np.array(all_targets), np.array(all_attns)


def compute_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> Dict[str, Any]:
    """Calculate comprehensive cybersecurity and ML metrics."""
    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def plot_confusion_matrix(cm: np.ndarray, out_path=CONFUSION_MATRIX_PATH) -> None:
    """Plot and save publication-quality confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["BENIGN (0)", "RANSOMWARE (1)"],
        yticklabels=["BENIGN (0)", "RANSOMWARE (1)"],
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Confusion Matrix (Test Split)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix saved to: {out_path}")


def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, auc_val: float, out_path=ROC_CURVE_PATH) -> None:
    """Plot and save Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"Proposed Model (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] ROC curve saved to: {out_path}")


def plot_pr_curve(y_true: np.ndarray, y_probs: np.ndarray, out_path=PR_CURVE_PATH) -> None:
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(recall, precision, color="#27ae60", lw=2.5, label=f"Proposed Model (AP = {ap:.4f})")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Precision-Recall curve saved to: {out_path}")


def evaluate_model() -> Dict[str, Any]:
    """Execute complete evaluation workflow on held-out test data."""
    print("=" * 60)
    print("TEST DATASET MODEL EVALUATION")
    print("=" * 60)

    # 1. Load test data
    df = load_raw_dataset()
    train_df, val_df, test_df = split_sequences(df, seed=SEED)
    scaler = load_scaler(SCALER_PATH)

    _, _, test_loader, arrays = get_dataloaders(
        train_df, val_df, test_df, scaler, batch_size=BATCH_SIZE
    )

    # 2. Load model
    model = load_trained_model(MODEL_CHECKPOINT_PATH, device=DEVICE)

    # 3. Predict on test loader
    probs, targets, attns = get_predictions(model, test_loader, device=DEVICE)

    # 4. Metrics
    metrics = compute_metrics(targets, probs, threshold=DEFAULT_THRESHOLD)

    # 5. Save plots
    plot_confusion_matrix(metrics["confusion_matrix"])
    plot_roc_curve(targets, probs, metrics["roc_auc"])
    plot_pr_curve(targets, probs)

    # 6. Save text classification report
    y_pred = (probs >= DEFAULT_THRESHOLD).astype(int)
    clf_report = classification_report(
        targets, y_pred, target_names=["BENIGN", "RANSOMWARE"], digits=4
    )

    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("PROPOSED MODEL TEST CLASSIFICATION REPORT\n")
        f.write(f"Threshold: {DEFAULT_THRESHOLD} | Test Samples: {len(targets)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(clf_report)
        f.write("\nDetailed Metrics:\n")
        f.write(f"  • Accuracy:    {metrics['accuracy']:.4f}\n")
        f.write(f"  • Precision:   {metrics['precision']:.4f}\n")
        f.write(f"  • Recall:      {metrics['recall']:.4f}\n")
        f.write(f"  • F1-Score:    {metrics['f1']:.4f}\n")
        f.write(f"  • ROC-AUC:     {metrics['roc_auc']:.4f}\n")
        f.write(f"  • Specificity: {metrics['specificity']:.4f}\n")
        f.write(f"  • FPR:         {metrics['fpr']:.4f}\n")
        f.write(f"  • FNR:         {metrics['fnr']:.4f}\n")
        f.write(f"  • Confusion Matrix: TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}\n")

    print(f"[+] Classification report saved to: {CLASSIFICATION_REPORT_PATH}")
    print("\n" + clf_report)

    return metrics


if __name__ == "__main__":
    evaluate_model()
