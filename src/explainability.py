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
    SEQ_LEN,
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


FEATURE_READABLE_NAMES = {
    "file_write_rate": "File Write Rate (ops/s)",
    "file_rename_rate": "File Rename Rate (ops/s)",
    "file_delete_rate": "File Delete Rate (ops/s)",
    "entropy_delta": "Shannon Entropy Delta (Randomness)",
    "extension_change_rate": "Extension Rename Rate (.locked)",
    "process_create_rate": "Process Creation Rate (PIDs/s)",
    "shadow_copy_cmd_count": "Shadow Copy Deletion (vssadmin)",
    "backup_delete_count": "Backup Catalog Deletion (wbadmin)",
    "cpu_percent": "Host CPU Utilization (%)",
    "memory_percent": "Host Memory Utilization (%)",
    "network_conn_rate": "Active Network Sockets",
    "file_open_rate": "File Open Handles (/s)",
    "directory_traversal_rate": "Directory Traversal Speed",
    "suspicious_api_rate": "Suspicious Crypto API Calls",
    "unique_extensions": "Distinct File Extensions Targeted",
    "encrypted_file_ratio": "Encrypted Files Ratio",
    "failed_access_rate": "Access Denied Frequency",
    "admin_action_rate": "Elevated Privilege Actions",
    "bytes_written_mb": "Total Disk Write Throughput (MB)",
    "new_unknown_process_rate": "Unverified Process Spawns",
}


def compute_sample_attribution(
    model: torch.nn.Module,
    scaler: Any,
    input_tensor_or_array: Any,
) -> Dict[str, Any]:
    """
    Compute local feature attribution scores (SHAP-style) and temporal attention
    decomposition for an individual sequence. Explains *why* the model made its verdict.
    """
    import plotly.graph_objects as go
    model.eval()

    if isinstance(input_tensor_or_array, np.ndarray):
        if input_tensor_or_array.ndim == 2:
            raw_matrix = input_tensor_or_array.astype(np.float32)
        elif input_tensor_or_array.ndim == 3:
            raw_matrix = input_tensor_or_array[0].astype(np.float32)
        else:
            raw_matrix = np.zeros((SEQ_LEN, len(FEATURES)), dtype=np.float32)
    elif isinstance(input_tensor_or_array, pd.DataFrame):
        raw_matrix = input_tensor_or_array[FEATURES].to_numpy(dtype=np.float32)
        if len(raw_matrix) < SEQ_LEN:
            pad = np.repeat(raw_matrix[:1], SEQ_LEN - len(raw_matrix), axis=0)
            raw_matrix = np.vstack([pad, raw_matrix])
        else:
            raw_matrix = raw_matrix[-SEQ_LEN:]
    else:
        raw_matrix = np.zeros((SEQ_LEN, len(FEATURES)), dtype=np.float32)

    # Scale the sequence
    try:
        scaled = scaler.transform(raw_matrix)
    except Exception:
        scaled = raw_matrix

    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)

    logits, attn = model(x)
    prob = float(torch.sigmoid(logits).item())

    # Gradient computation for attribution
    model.zero_grad()
    logits.backward(retain_graph=True)
    grad = x.grad.data.cpu().numpy()[0]  # (20, 20)
    x_val = x.data.cpu().numpy()[0]

    # Input * Gradient attribution per timestep
    temporal_attr = grad * x_val  # (20, 20)
    attn_weights = attn.squeeze(0).detach().cpu().numpy() if attn is not None else np.ones(SEQ_LEN) / SEQ_LEN

    # Combine temporal attention with gradient attribution: feature_attribution = sum_t (attn[t] * temporal_attr[t, f])
    feature_impacts = np.zeros(len(FEATURES), dtype=np.float32)
    for t in range(SEQ_LEN):
        feature_impacts += attn_weights[t] * temporal_attr[t]

    abs_sum = np.sum(np.abs(feature_impacts))
    if abs_sum > 1e-6:
        pct_contributions = (feature_impacts / abs_sum) * 100.0
    else:
        latest_vec = raw_matrix[-1]
        norm = np.sum(np.abs(latest_vec)) + 1e-6
        pct_contributions = (latest_vec / norm) * 100.0

    contributions_list = []
    for idx, feat_name in enumerate(FEATURES):
        val = float(raw_matrix[-1, idx])
        impact = float(pct_contributions[idx])
        direction = "MALICIOUS" if impact > 0 else "BENIGN"
        readable = FEATURE_READABLE_NAMES.get(feat_name, feat_name)
        contributions_list.append({
            "feature": feat_name,
            "readable_name": readable,
            "raw_value": val,
            "contribution_pct": round(impact, 2),
            "abs_impact": abs(round(impact, 2)),
            "direction": direction,
        })

    contributions_list.sort(key=lambda item: item["abs_impact"], reverse=True)

    top_pos = [c for c in contributions_list if c["direction"] == "MALICIOUS"]
    if top_pos and prob >= 0.50:
        primary = top_pos[0]
        secondary = top_pos[1] if len(top_pos) > 1 else None
        smoking_gun = (
            f"Primary trigger: {primary['readable_name']} contributed {primary['contribution_pct']:+.1f}% "
            f"towards the ransomware verdict (measured value: {primary['raw_value']:.2f})."
        )
        if secondary:
            smoking_gun += f" Corroborated by {secondary['readable_name']} ({secondary['contribution_pct']:+.1f}%)."
    else:
        smoking_gun = "Workload metrics remain within nominal operational boundaries. No anomalous attack trigger detected."

    return {
        "probability": round(prob * 100.0, 2),
        "risk_level": "CRITICAL" if prob >= 0.90 else ("HIGH" if prob >= 0.70 else ("MEDIUM" if prob >= 0.30 else "LOW")),
        "contributions": contributions_list,
        "top_features": contributions_list[:5],
        "attention_weights": [round(float(w), 4) for w in attn_weights],
        "smoking_gun": smoking_gun,
        "max_attention_timestep": int(np.argmax(attn_weights)),
    }


def create_attribution_bar_chart(explain_result: Dict[str, Any], top_n: int = 8):
    """Generate interactive Plotly horizontal bar chart of feature contributions."""
    import plotly.graph_objects as go
    top_items = explain_result["contributions"][:top_n]
    top_items = list(reversed(top_items))

    names = [item["readable_name"] for item in top_items]
    impacts = [item["contribution_pct"] for item in top_items]
    colors = ["#ef4444" if val > 0 else "#10b981" for val in impacts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=impacts,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
        text=[f"{val:+.1f}%" for val in impacts],
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>Attribution Force: %{x:+.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=f"Feature Attribution Decomposition (Model Probability: {explain_result['probability']}%)",
        xaxis_title="Contribution to Ransomware Likelihood (%)",
        yaxis_title="",
        template="plotly_dark",
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


if __name__ == "__main__":
    run_explainability()

