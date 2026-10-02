"""
Unit Tests: Preprocessing & Data Leakage Prevention
"""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURES,
    SEQ_LEN,
    N_FEATURES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SCALER_PATH,
)
from src.preprocessing import (
    load_raw_dataset,
    split_sequences,
    fit_scaler,
    load_scaler,
    transform_features,
    create_3d_sequences,
    create_tabular_sequence_features,
)


def test_split_sequences_no_leakage():
    """Verify that sequences are split strictly at sequence level with zero overlap."""
    df = load_raw_dataset()
    train_df, val_df, test_df = split_sequences(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)

    train_seqs = set(train_df["sequence_id"].unique())
    val_seqs = set(val_df["sequence_id"].unique())
    test_seqs = set(test_df["sequence_id"].unique())

    # Ensure mutually disjoint sets
    assert len(train_seqs & val_seqs) == 0, "Train and Val sequence IDs overlap!"
    assert len(train_seqs & test_seqs) == 0, "Train and Test sequence IDs overlap!"
    assert len(val_seqs & test_seqs) == 0, "Val and Test sequence IDs overlap!"

    # Ensure total sequences match
    total_seqs = df["sequence_id"].nunique()
    assert len(train_seqs) + len(val_seqs) + len(test_seqs) == total_seqs


def test_scaler_fitting_on_train_only():
    """Verify scaler is fitted on training telemetry and successfully saved/loaded."""
    df = load_raw_dataset()
    train_df, val_df, test_df = split_sequences(df)
    scaler = fit_scaler(train_df)

    assert hasattr(scaler, "mean_")
    assert len(scaler.mean_) == len(FEATURES)

    loaded_scaler = load_scaler(SCALER_PATH)
    np.testing.assert_allclose(scaler.mean_, loaded_scaler.mean_)


def test_create_3d_sequences_dimensions():
    """Verify conversion from 2D DataFrame to 3D tensor matches (N, 20, 20)."""
    df = load_raw_dataset()
    train_df, _, _ = split_sequences(df)
    scaler = load_scaler(SCALER_PATH)

    scaled = transform_features(train_df, scaler)
    X, y, seq_ids = create_3d_sequences(train_df, scaled, seq_len=SEQ_LEN)

    n_train_seqs = train_df["sequence_id"].nunique()
    assert X.shape == (n_train_seqs, SEQ_LEN, N_FEATURES)
    assert y.shape == (n_train_seqs,)
    assert len(seq_ids) == n_train_seqs


def test_tabular_features_dimensions():
    """Verify tabular feature extraction shape for baseline models."""
    sample_3d = np.random.randn(50, SEQ_LEN, N_FEATURES)
    tab_feat = create_tabular_sequence_features(sample_3d)
    # 6 statistics per feature: mean, std, min, max, last, delta
    expected_dim = N_FEATURES * 6
    assert tab_feat.shape == (50, expected_dim)
