"""
Unit Tests: Input Validation & Error Handling
"""

import pytest
import pandas as pd
import numpy as np

from src.config import FEATURES, SEQ_LEN
from src.preprocessing import prepare_single_sequence_for_inference, load_scaler
from src.predict import predict_csv


@pytest.fixture
def valid_sample_df():
    """Create a minimal valid sequence DataFrame with 20 rows and 20 features."""
    data = {f: np.random.uniform(0.1, 10.0, SEQ_LEN) for f in FEATURES}
    data["time_step"] = list(range(SEQ_LEN))
    data["sequence_id"] = [1] * SEQ_LEN
    return pd.DataFrame(data)


def test_validation_missing_columns(valid_sample_df):
    """Verify ValueError is raised if any required feature column is missing."""
    corrupted_df = valid_sample_df.drop(columns=["entropy_delta", "file_write_rate"])
    with pytest.raises(ValueError, match="missing required feature columns"):
        predict_csv(corrupted_df)


def test_validation_insufficient_rows(valid_sample_df):
    """Verify ValueError is raised if sequence has fewer than 20 timesteps."""
    short_df = valid_sample_df.iloc[:10]  # Only 10 rows
    with pytest.raises(ValueError, match="Minimum required is 20 timesteps"):
        predict_csv(short_df)


def test_validation_nan_values(valid_sample_df):
    """Verify ValueError is raised if feature columns contain NaN values."""
    nan_df = valid_sample_df.copy()
    nan_df.loc[5, "file_write_rate"] = np.nan
    with pytest.raises(ValueError, match="null / NaN values"):
        predict_csv(nan_df)


def test_validation_nonexistent_file():
    """Verify FileNotFoundError is raised if CSV file does not exist."""
    with pytest.raises(FileNotFoundError):
        predict_csv("non_existent_file_path_12345.csv")
