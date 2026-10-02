"""
Unit Tests: Dataset Generation & Validation
"""

import pandas as pd
import numpy as np
import pytest
from src.config import (
    N_SEQUENCES,
    SEQ_LEN,
    N_FEATURES,
    FEATURES,
    METADATA_COLUMNS,
    DATASET_PATH,
)
from src.dataset import (
    generate_synthetic_dataset,
    validate_dataset,
    generate_host_profiles,
)


def test_dataset_file_exists():
    """Verify generated dataset file exists."""
    assert DATASET_PATH.exists(), f"Dataset file does not exist at {DATASET_PATH}"


def test_dataset_shape_and_columns():
    """Verify total rows, sequence count, and columns in dataset."""
    df = pd.read_csv(DATASET_PATH)
    expected_rows = N_SEQUENCES * SEQ_LEN
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"

    expected_cols = set(METADATA_COLUMNS + FEATURES)
    assert set(df.columns) == expected_cols, f"Mismatch in expected columns: {set(df.columns) ^ expected_cols}"


def test_dataset_classes_and_balance():
    """Verify balanced distribution between Benign (0) and Ransomware (1)."""
    df = pd.read_csv(DATASET_PATH)
    seq_labels = df.groupby("sequence_id")["label"].first()

    assert set(seq_labels.unique()) == {0, 1}
    assert seq_labels.value_counts()[0] == N_SEQUENCES // 2
    assert seq_labels.value_counts()[1] == N_SEQUENCES // 2


def test_dataset_no_missing_values():
    """Verify zero missing values in generated dataset."""
    df = pd.read_csv(DATASET_PATH)
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected NaN values"


def test_dataset_timesteps_continuity():
    """Verify every sequence has continuous timesteps from 0 to 19."""
    df = pd.read_csv(DATASET_PATH)
    timesteps_per_seq = df.groupby("sequence_id")["time_step"].apply(list)
    expected_timesteps = list(range(SEQ_LEN))

    for s_id, steps in timesteps_per_seq.items():
        assert steps == expected_timesteps, f"Sequence {s_id} timesteps broken"


def test_small_generator_run():
    """Test generating a small dynamic dataset in memory."""
    small_df = generate_synthetic_dataset(n_sequences=20, seed=123)
    assert len(small_df) == 20 * SEQ_LEN
    audit = validate_dataset(small_df)
    assert audit["missing_values"] == 0
    assert audit["n_sequences"] == 20
