"""
Unit Tests: PyTorch Model Architectures & Shapes
"""

import pytest
import torch

from src.config import N_FEATURES, SEQ_LEN
from src.model import (
    RansomwareHybridModel,
    CNNOnlyModel,
    BiLSTMOnlyModel,
    CNN_BiLSTMModel,
)


def test_proposed_hybrid_model_shapes():
    """Verify proposed hybrid model output shapes for (batch, 20, 20) input."""
    batch_size = 8
    model = RansomwareHybridModel(in_features=N_FEATURES)
    model.eval()

    dummy_input = torch.randn(batch_size, SEQ_LEN, N_FEATURES)
    logits, attn_weights = model(dummy_input)

    assert logits.shape == (batch_size, 1), f"Expected logits shape {(batch_size, 1)}, got {logits.shape}"
    assert attn_weights.shape == (batch_size, SEQ_LEN), f"Expected attention weights shape {(batch_size, SEQ_LEN)}, got {attn_weights.shape}"

    # Attention weights should sum to 1.0 across the sequence dimension
    attn_sums = attn_weights.sum(dim=-1).detach().numpy()
    for s in attn_sums:
        assert pytest.approx(1.0, abs=1e-5) == s


def test_cnn_only_model_shape():
    """Verify CNN baseline output shape."""
    batch_size = 4
    model = CNNOnlyModel(in_features=N_FEATURES)
    dummy_input = torch.randn(batch_size, SEQ_LEN, N_FEATURES)
    logits, attn = model(dummy_input)

    assert logits.shape == (batch_size, 1)
    assert attn is None


def test_bilstm_only_model_shape():
    """Verify BiLSTM baseline output shape."""
    batch_size = 4
    model = BiLSTMOnlyModel(in_features=N_FEATURES)
    dummy_input = torch.randn(batch_size, SEQ_LEN, N_FEATURES)
    logits, attn = model(dummy_input)

    assert logits.shape == (batch_size, 1)
    assert attn is None


def test_cnn_bilstm_model_shape():
    """Verify CNN+BiLSTM baseline output shape."""
    batch_size = 4
    model = CNN_BiLSTMModel(in_features=N_FEATURES)
    dummy_input = torch.randn(batch_size, SEQ_LEN, N_FEATURES)
    logits, attn = model(dummy_input)

    assert logits.shape == (batch_size, 1)
    assert attn is None
