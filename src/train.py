"""
Model Training & Checkpointing Module
Trains the Proposed Hybrid Model (1D-CNN + BiLSTM + Attention) with
early stopping, gradient clipping, experiment metadata tracking, and loss/accuracy curves.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.config import (
    SEED,
    SEQ_LEN,
    N_FEATURES,
    FEATURES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP,
    DEVICE,
    MODEL_CHECKPOINT_PATH,
    SCALER_PATH,
    TRAINING_HISTORY_PATH,
    TRAINING_CURVE_PATH,
    EXPERIMENTS_DIR,
    CNN_FILTERS,
    LSTM_HIDDEN_SIZE,
    LSTM_LAYERS,
    DENSE_UNITS,
    DROPOUT_RATE,
)
from src.dataset import set_seed
from src.preprocessing import (
    load_raw_dataset,
    split_sequences,
    fit_scaler,
    get_dataloaders,
)
from src.model import RansomwareHybridModel


def set_reproducibility(seed: int = SEED) -> None:
    """Ensure strict multi-library deterministic reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = GRADIENT_CLIP,
) -> Tuple[float, float]:
    """Train model for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits, _ = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate model on validation or test DataLoader. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def plot_training_curves(history: Dict[str, List[float]]) -> None:
    """Generate and save training vs validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Loss Curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#3498db", linewidth=2.2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#e74c3c", linewidth=2.2, linestyle="--")
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy Curve
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="#2ecc71", linewidth=2.2)
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#9b59b6", linewidth=2.2, linestyle="--")
    ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.suptitle("Proposed Model (1D-CNN + BiLSTM + Attention) Training Dynamics", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(TRAINING_CURVE_PATH, dpi=300)
    plt.close()
    print(f"[+] Training curves saved to: {TRAINING_CURVE_PATH}")


def train_hybrid_model() -> Dict[str, Any]:
    """Execute end-to-end training pipeline for proposed hybrid model."""
    print("=" * 60)
    print("PROPOSED HYBRID MODEL TRAINING PIPELINE")
    print(f"Device: {DEVICE} | Seed: {SEED} | Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}")
    print("=" * 60)

    set_reproducibility(SEED)

    # 1. Load raw dataset and split by sequence (preventing data leakage)
    df = load_raw_dataset()
    train_df, val_df, test_df = split_sequences(df, seed=SEED)

    # 2. Fit StandardScaler strictly on training split
    scaler = fit_scaler(train_df, save_path=SCALER_PATH)

    # 3. Create PyTorch DataLoaders
    train_loader, val_loader, test_loader, arrays = get_dataloaders(
        train_df, val_df, test_df, scaler, batch_size=BATCH_SIZE
    )

    # 4. Instantiate Proposed Hybrid Model
    model = RansomwareHybridModel(
        in_features=N_FEATURES,
        cnn_filters=CNN_FILTERS,
        lstm_hidden=LSTM_HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        dense_units=DENSE_UNITS,
        dropout=DROPOUT_RATE,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training state
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    start_time = time.time()

    print(f"[*] Training started ({EPOCHS} epochs max, early stopping patience={EARLY_STOPPING_PATIENCE})...\n")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, DEVICE)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} [{epoch_time:.1f}s] - "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}",
            end="",
        )

        # Early stopping & model checkpointing on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print("  --> Best Model Saved [*]")
        else:
            patience_counter += 1
            print(f"  (patience {patience_counter}/{EARLY_STOPPING_PATIENCE})")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n[!] Early stopping triggered at epoch {epoch} (no validation improvement for {EARLY_STOPPING_PATIENCE} epochs).")
                break

    total_training_time = time.time() - start_time
    print(f"\n[+] Training finished in {total_training_time:.1f}s. Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f})")

    # Save best checkpoint
    checkpoint = {
        "model_state_dict": best_model_state,
        "architecture_params": {
            "in_features": N_FEATURES,
            "cnn_filters": CNN_FILTERS,
            "lstm_hidden": LSTM_HIDDEN_SIZE,
            "lstm_layers": LSTM_LAYERS,
            "dense_units": DENSE_UNITS,
            "dropout": DROPOUT_RATE,
        },
        "sequence_length": SEQ_LEN,
        "feature_count": N_FEATURES,
        "feature_names": FEATURES,
        "training_config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "gradient_clip": GRADIENT_CLIP,
        },
        "random_seed": SEED,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
    }
    torch.save(checkpoint, MODEL_CHECKPOINT_PATH)
    print(f"[+] Model checkpoint saved to: {MODEL_CHECKPOINT_PATH}")

    # Save training history CSV
    history_df = pd.DataFrame(history)
    history_df.index.name = "epoch"
    history_df.to_csv(TRAINING_HISTORY_PATH)
    print(f"[+] Training history saved to: {TRAINING_HISTORY_PATH}")

    # Generate training curves plot
    plot_training_curves(history)

    # Save experiment tracking JSON
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_metadata = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "dataset_size": len(df),
        "sequence_length": SEQ_LEN,
        "features": N_FEATURES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "architecture": "CNN-BiLSTM-Attention",
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_accuracy": float(best_val_acc),
        "training_time_seconds": round(total_training_time, 2),
    }
    exp_file = EXPERIMENTS_DIR / f"{experiment_id}.json"
    with open(exp_file, "w") as f:
        json.dump(exp_metadata, f, indent=4)
    print(f"[+] Experiment metadata logged to: {exp_file}")

    return checkpoint


if __name__ == "__main__":
    train_hybrid_model()
