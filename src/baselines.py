"""
Baseline Models Benchmark Module
Trains and evaluates baseline models on the exact same test split:
  1. Logistic Regression (Statistical sequence features)
  2. Random Forest (Statistical sequence features)
  3. CNN Only
  4. BiLSTM Only
  5. CNN + BiLSTM (Without Attention)
  6. Proposed Hybrid Model (CNN + BiLSTM + Attention)
Saves results to reports/model_comparison.csv.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.config import (
    SEED,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    MODEL_COMPARISON_PATH,
    MODEL_CHECKPOINT_PATH,
    SCALER_PATH,
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
    create_tabular_sequence_features,
)
from src.model import (
    CNNOnlyModel,
    BiLSTMOnlyModel,
    CNN_BiLSTMModel,
    RansomwareHybridModel,
)
from src.evaluate import load_trained_model, get_predictions, compute_metrics


def train_eval_torch_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int = 15,
    device: str = DEVICE,
) -> Dict[str, float]:
    """Train a baseline PyTorch neural network and evaluate on test loader."""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        # Val evaluation
        model.eval()
        val_loss = 0.0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                logits, _ = model(X_batch)
                val_loss += criterion(logits, y_batch).item() * X_batch.size(0)
                total += X_batch.size(0)

        val_loss /= total
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict().copy()

    # Load best weights and evaluate on test set
    if best_weights is not None:
        model.load_state_dict(best_weights)

    probs, targets, _ = get_predictions(model, test_loader, device=device)
    metrics = compute_metrics(targets, probs)
    return {
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
        "ROC-AUC": metrics["roc_auc"],
    }


def run_baselines_benchmark() -> pd.DataFrame:
    """Execute end-to-end benchmark comparison across all 6 models."""
    print("=" * 60)
    print("BENCHMARKING BASELINE & PROPOSED MODELS")
    print("=" * 60)

    # 1. Load data & partitions
    df = load_raw_dataset()
    train_df, val_df, test_df = split_sequences(df, seed=SEED)
    scaler = load_scaler(SCALER_PATH)

    train_loader, val_loader, test_loader, arrays = get_dataloaders(
        train_df, val_df, test_df, scaler, batch_size=BATCH_SIZE
    )

    X_train_3d = arrays["X_train"]
    y_train = arrays["y_train"]
    X_test_3d = arrays["X_test"]
    y_test = arrays["y_test"]

    # Generate tabular features for traditional ML baselines
    X_train_tab = create_tabular_sequence_features(X_train_3d)
    X_test_tab = create_tabular_sequence_features(X_test_3d)

    results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. Baseline: Logistic Regression
    # -------------------------------------------------------------
    print("[*] Evaluating Baseline 1: Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_train_tab, y_train)
    lr_probs = lr.predict_proba(X_test_tab)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)

    results.append({
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y_test, lr_preds),
        "Precision": precision_score(y_test, lr_preds, zero_division=0),
        "Recall": recall_score(y_test, lr_preds, zero_division=0),
        "F1": f1_score(y_test, lr_preds, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, lr_probs),
    })

    # -------------------------------------------------------------
    # 2. Baseline: Random Forest
    # -------------------------------------------------------------
    print("[*] Evaluating Baseline 2: Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_tab, y_train)
    rf_probs = rf.predict_proba(X_test_tab)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)

    results.append({
        "Model": "Random Forest",
        "Accuracy": accuracy_score(y_test, rf_preds),
        "Precision": precision_score(y_test, rf_preds, zero_division=0),
        "Recall": recall_score(y_test, rf_preds, zero_division=0),
        "F1": f1_score(y_test, rf_preds, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, rf_probs),
    })

    # -------------------------------------------------------------
    # 3. Baseline: CNN Only
    # -------------------------------------------------------------
    print("[*] Evaluating Baseline 3: CNN Only...")
    cnn_model = CNNOnlyModel(in_features=N_FEATURES, cnn_filters=CNN_FILTERS)
    cnn_res = train_eval_torch_model(cnn_model, train_loader, val_loader, test_loader, epochs=12)
    cnn_res["Model"] = "CNN"
    results.append(cnn_res)

    # -------------------------------------------------------------
    # 4. Baseline: BiLSTM Only
    # -------------------------------------------------------------
    print("[*] Evaluating Baseline 4: BiLSTM Only...")
    bilstm_model = BiLSTMOnlyModel(in_features=N_FEATURES, lstm_hidden=LSTM_HIDDEN_SIZE)
    bilstm_res = train_eval_torch_model(bilstm_model, train_loader, val_loader, test_loader, epochs=12)
    bilstm_res["Model"] = "BiLSTM"
    results.append(bilstm_res)

    # -------------------------------------------------------------
    # 5. Baseline: CNN + BiLSTM
    # -------------------------------------------------------------
    print("[*] Evaluating Baseline 5: CNN + BiLSTM...")
    cnn_lstm_model = CNN_BiLSTMModel(in_features=N_FEATURES, cnn_filters=CNN_FILTERS, lstm_hidden=LSTM_HIDDEN_SIZE)
    cnn_lstm_res = train_eval_torch_model(cnn_lstm_model, train_loader, val_loader, test_loader, epochs=12)
    cnn_lstm_res["Model"] = "CNN + BiLSTM"
    results.append(cnn_lstm_res)

    # -------------------------------------------------------------
    # 6. Proposed Hybrid Model (Loaded from checkpoint)
    # -------------------------------------------------------------
    print("[*] Evaluating Proposed Hybrid Model (1D-CNN + BiLSTM + Attention)...")
    try:
        proposed_model = load_trained_model(MODEL_CHECKPOINT_PATH, device=DEVICE)
        prop_probs, prop_targets, _ = get_predictions(proposed_model, test_loader, device=DEVICE)
        prop_metrics = compute_metrics(prop_targets, prop_probs)
        results.append({
            "Model": "Proposed Hybrid",
            "Accuracy": prop_metrics["accuracy"],
            "Precision": prop_metrics["precision"],
            "Recall": prop_metrics["recall"],
            "F1": prop_metrics["f1"],
            "ROC-AUC": prop_metrics["roc_auc"],
        })
    except Exception as e:
        print(f"[!] Warning: Could not load trained checkpoint ({e}), evaluating freshly trained model.")
        prop_model = RansomwareHybridModel().to(DEVICE)
        prop_res = train_eval_torch_model(prop_model, train_loader, val_loader, test_loader, epochs=12)
        prop_res["Model"] = "Proposed Hybrid"
        results.append(prop_res)

    comparison_df = pd.DataFrame(results)
    # Reorder columns
    comparison_df = comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]

    # Round actual metrics
    for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        comparison_df[col] = comparison_df[col].round(4)

    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"\n[+] Model comparison table saved to: {MODEL_COMPARISON_PATH}\n")
    print(comparison_df.to_string(index=False))

    return comparison_df


if __name__ == "__main__":
    run_baselines_benchmark()
