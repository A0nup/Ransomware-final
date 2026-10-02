"""
Explainability & Interpretability Module
Extracts attention weights across timesteps and calculates
permutation feature importance on the held-out test sequences.
Saves reports/feature_importance.csv and reports/feature_importance.png.
"""

from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score, f1_score

from src.config import (
    MODEL_CHECKPOINT_PATH,
    SCALER_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_IMPORTANCE_PLOT_PATH,
    FEATURES,
    SEED,
    BATCH_SIZE,
    DEVICE,
)
from src.preprocessing import (
    load_raw_dataset,
    split_sequences,
    load_scaler,
    get_dataloaders,
)
from src.evaluate import load_trained_model, get_predictions


def extract_sample_attention(
    model: torch.nn.Module, sample_tensor: torch.Tensor, device: str = DEVICE
) -> np.ndarray:
    """
    Extract attention weights for a single sequence tensor (1, 20, 20).
    Returns a 1D numpy array of shape (20,) representing attention weights.
    """
    model.eval()
    with torch.no_grad():
        sample_tensor = sample_tensor.to(device)
        _, attn_weights = model(sample_tensor)
        if attn_weights is not None:
            return attn_weights.squeeze(0).cpu().numpy()
        return np.ones(sample_tensor.size(1)) / sample_tensor.size(1)


def compute_permutation_feature_importance(
    model: torch.nn.Module,
    X_test_3d: np.ndarray,
    y_test: np.ndarray,
    device: str = DEVICE,
    n_repeats: int = 3,
) -> pd.DataFrame:
    """
    Compute permutation feature importance on 3D sequence test data.
    Estimates the drop in model discrimination (ROC-AUC) when each feature is shuffled.
    """
    model.eval()
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test_3d, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Baseline performance
    base_probs, base_targets, _ = get_predictions(model, loader, device=device)
    base_auc = roc_auc_score(base_targets, base_probs)

    n_features = len(FEATURES)
    importance_scores = []

    print(f"[*] Baseline Test ROC-AUC: {base_auc:.4f}. Permuting {n_features} features...")

    rng = np.random.RandomState(SEED)

    for f_idx, feat_name in enumerate(FEATURES):
        perm_auc_drops = []

        for _ in range(n_repeats):
            X_permuted = X_test_3d.copy()
            # Permute feature values across all sequences and timesteps
            flat_feat = X_permuted[:, :, f_idx].flatten()
            rng.shuffle(flat_feat)
            X_permuted[:, :, f_idx] = flat_feat.reshape(X_permuted.shape[0], X_permuted.shape[1])

            perm_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_permuted, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
            )
            perm_loader = torch.utils.data.DataLoader(perm_dataset, batch_size=BATCH_SIZE, shuffle=False)

            p_probs, p_targets, _ = get_predictions(model, perm_loader, device=device)
            p_auc = roc_auc_score(p_targets, p_probs)
            perm_auc_drops.append(max(0.0, base_auc - p_auc))

        mean_drop = np.mean(perm_auc_drops)
        std_drop = np.std(perm_auc_drops)

        importance_scores.append({
            "Feature": feat_name,
            "Mean_AUC_Drop": round(float(mean_drop), 5),
            "Std_AUC_Drop": round(float(std_drop), 5),
        })

    imp_df = pd.DataFrame(importance_scores)
    imp_df = imp_df.sort_values(by="Mean_AUC_Drop", ascending=False).reset_index(drop=True)
    return imp_df


def plot_feature_importance(imp_df: pd.DataFrame, out_path=FEATURE_IMPORTANCE_PLOT_PATH) -> None:
    """Generate and save horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Plot top features
    y_pos = np.arange(len(imp_df))
    ax.barh(
        y_pos,
        imp_df["Mean_AUC_Drop"],
        xerr=imp_df["Std_AUC_Drop"],
        align="center",
        color="#3498db",
        edgecolor="#2980b9",
        alpha=0.85,
        capsize=3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_df["Feature"], fontsize=10)
    ax.invert_yaxis()  # Highest importance at the top
    ax.set_xlabel("Mean ROC-AUC Drop Upon Permutation", fontsize=11, fontweight="bold")
    ax.set_title("Permutation Feature Importance (Proposed Model)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Feature importance plot saved to: {out_path}")


def run_explainability() -> pd.DataFrame:
    """Execute complete explainability pipeline."""
    print("=" * 60)
    print("MODEL EXPLAINABILITY & PERMUTATION IMPORTANCE")
    print("=" * 60)

    df = load_raw_dataset()
    train_df, val_df, test_df = split_sequences(df, seed=SEED)
    scaler = load_scaler(SCALER_PATH)

    _, _, _, arrays = get_dataloaders(
        train_df, val_df, test_df, scaler, batch_size=BATCH_SIZE
    )

    model = load_trained_model(MODEL_CHECKPOINT_PATH, device=DEVICE)

    imp_df = compute_permutation_feature_importance(
        model, arrays["X_test"], arrays["y_test"], device=DEVICE
    )

    imp_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    print(f"[+] Feature importance table saved to: {FEATURE_IMPORTANCE_PATH}\n")
    print(imp_df.head(10).to_string(index=False))

    plot_feature_importance(imp_df)

    return imp_df


if __name__ == "__main__":
    run_explainability()
