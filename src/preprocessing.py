"""
Data Preprocessing & Sequence Engineering Module
Enforces strict sequence-level splitting to prevent data leakage.
Fits standard scaling strictly on training sequences only.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURES,
    METADATA_COLUMNS,
    SEQ_LEN,
    N_FEATURES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
    BATCH_SIZE,
    SCALER_PATH,
    DATASET_PATH,
)


def load_raw_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load and sort telemetry dataset by sequence_id and time_step."""
    path = csv_path or DATASET_PATH
    df = pd.read_csv(path)
    df = df.sort_values(by=["sequence_id", "time_step"]).reset_index(drop=True)
    return df


def split_sequences(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset strictly at the SEQUENCE level to prevent data leakage.
    Rows from the same sequence never span multiple partitions.
    Uses stratified splitting based on sequence class labels.
    """
    # Extract unique sequences with their ground-truth labels
    seq_meta = (
        df.groupby("sequence_id")
        .agg({"label": "first", "host_id": "first"})
        .reset_index()
    )

    # 1. Split Train vs (Val + Test)
    train_seqs, temp_seqs = train_test_split(
        seq_meta,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=seq_meta["label"],
    )

    # 2. Split Val vs Test
    val_prop_of_temp = val_ratio / (val_ratio + test_ratio)
    val_seqs, test_seqs = train_test_split(
        temp_seqs,
        test_size=(1.0 - val_prop_of_temp),
        random_state=seed,
        stratify=temp_seqs["label"],
    )

    train_df = (
        df[df["sequence_id"].isin(train_seqs["sequence_id"])]
        .sort_values(by=["sequence_id", "time_step"])
        .reset_index(drop=True)
    )
    val_df = (
        df[df["sequence_id"].isin(val_seqs["sequence_id"])]
        .sort_values(by=["sequence_id", "time_step"])
        .reset_index(drop=True)
    )
    test_df = (
        df[df["sequence_id"].isin(test_seqs["sequence_id"])]
        .sort_values(by=["sequence_id", "time_step"])
        .reset_index(drop=True)
    )

    print(
        f"[+] Sequence Split: Train={train_seqs.shape[0]} seqs ({train_df.shape[0]} rows), "
        f"Val={val_seqs.shape[0]} seqs ({val_df.shape[0]} rows), "
        f"Test={test_seqs.shape[0]} seqs ({test_df.shape[0]} rows)"
    )

    return train_df, val_df, test_df


def fit_scaler(train_df: pd.DataFrame, save_path: Optional[str] = None) -> StandardScaler:
    """
    Fit StandardScaler strictly on training telemetry features.
    Saves scaler artifact for evaluation and live inference.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES].values)

    target_path = save_path or SCALER_PATH
    joblib.dump(scaler, target_path)
    print(f"[+] Scaler fitted on training data and saved to: {target_path}")
    return scaler


def load_scaler(scaler_path: Optional[str] = None) -> StandardScaler:
    """Load saved StandardScaler artifact."""
    target_path = scaler_path or SCALER_PATH
    return joblib.load(target_path)


def transform_features(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Apply fitted scaler transformation to features."""
    return scaler.transform(df[FEATURES].values)


def create_3d_sequences(
    df: pd.DataFrame, scaled_features: np.ndarray, seq_len: int = SEQ_LEN
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct 3D tensors of shape (N_sequences, seq_len, n_features)
    and corresponding 1D label array.
    """
    df_sorted = df.sort_values(by=["sequence_id", "time_step"]).reset_index(drop=True)
    unique_seq_ids = df_sorted["sequence_id"].unique()
    n_seqs = len(unique_seq_ids)

    if len(df_sorted) != n_seqs * seq_len:
        raise ValueError(
            f"Total rows ({len(df_sorted)}) does not match expected {n_seqs} * {seq_len}"
        )

    # Direct reshape on sorted rows
    X = scaled_features.reshape(n_seqs, seq_len, len(FEATURES)).astype(np.float32)
    y = (
        df_sorted.groupby("sequence_id", sort=False)["label"]
        .first()
        .values.astype(np.float32)
    )
    seq_ids = (
        df_sorted.groupby("sequence_id", sort=False)["sequence_id"]
        .first()
        .values.astype(np.int64)
    )

    return X, y, seq_ids


def create_tabular_sequence_features(X_3d: np.ndarray) -> np.ndarray:
    """
    Extract aggregated statistical sequence features for traditional ML baselines
    (Logistic Regression, Random Forest).
    Features computed across time dimension: mean, std, min, max, last_step, delta.
    Shape: (N_sequences, N_features * 6)
    """
    mean_feat = np.mean(X_3d, axis=1)  # (N, 20)
    std_feat = np.std(X_3d, axis=1)    # (N, 20)
    min_feat = np.min(X_3d, axis=1)    # (N, 20)
    max_feat = np.max(X_3d, axis=1)    # (N, 20)
    last_feat = X_3d[:, -1, :]         # (N, 20)
    delta_feat = X_3d[:, -1, :] - X_3d[:, 0, :]  # (N, 20)

    tabular_features = np.hstack(
        [mean_feat, std_feat, min_feat, max_feat, last_feat, delta_feat]
    )
    return tabular_features


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: StandardScaler,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Prepare PyTorch DataLoaders for Train, Validation, and Test partitions.
    """
    # Transform partitions
    scaled_train = transform_features(train_df, scaler)
    scaled_val = transform_features(val_df, scaler)
    scaled_test = transform_features(test_df, scaler)

    # Build 3D sequences
    X_train, y_train, train_ids = create_3d_sequences(train_df, scaled_train)
    X_val, y_val, val_ids = create_3d_sequences(val_df, scaled_val)
    X_test, y_test, test_ids = create_3d_sequences(test_df, scaled_test)

    # Create PyTorch datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    # DataLoaders (Shuffle training set only)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    arrays = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }

    return train_loader, val_loader, test_loader, arrays


def prepare_single_sequence_for_inference(
    df: pd.DataFrame, scaler: StandardScaler, seq_len: int = SEQ_LEN
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Validate, sort, scale, and format an uploaded or sample single sequence DataFrame
    into a PyTorch tensor (1, seq_len, n_features) for live model inference.
    """
    # Verify required feature columns
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Uploaded data missing required feature columns: {missing_features}")

    # Check for NaN or Inf
    if df[FEATURES].isnull().any().any():
        raise ValueError("Uploaded data contains null / NaN values in feature columns.")

    # Filter to single sequence if multiple sequences exist in the DataFrame
    if "sequence_id" in df.columns and df["sequence_id"].nunique() > 1:
        first_seq_id = df["sequence_id"].iloc[0]
        df = df[df["sequence_id"] == first_seq_id].copy()

    # Sort if time_step column exists
    if "time_step" in df.columns:
        df_sorted = df.sort_values(by="time_step").reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    if len(df_sorted) < seq_len:
        raise ValueError(
            f"Uploaded sequence has only {len(df_sorted)} rows. Minimum required is {seq_len} timesteps."
        )

    # Take the first/last seq_len rows if longer
    df_window = df_sorted.iloc[:seq_len].copy()

    # Scale features
    scaled_vals = scaler.transform(df_window[FEATURES].values)
    tensor = torch.tensor(scaled_vals, dtype=torch.float32).unsqueeze(0)  # (1, 20, 20)

    return tensor, df_window
