"""
Neural Network Architectures Module
Defines Proposed Hybrid Model (1D-CNN + BiLSTM + Attention)
and baseline variants for ablation and comparative evaluation.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    N_FEATURES,
    SEQ_LEN,
    CNN_FILTERS,
    CNN_KERNEL_SIZE,
    LSTM_HIDDEN_SIZE,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    DENSE_UNITS,
    DROPOUT_RATE,
)


class Conv1DFeatureExtractor(nn.Module):
    """
    1D Convolutional Neural Network block for local temporal pattern extraction.
    Input: (batch, seq_len, in_channels)
    Output: (batch, seq_len, out_channels)
    """

    def __init__(
        self,
        in_channels: int = N_FEATURES,
        out_channels: int = CNN_FILTERS,
        kernel_size: int = CNN_KERNEL_SIZE,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Conv1d expects (batch, in_channels, seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose from (batch, seq_len, features) to (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.relu2(self.bn2(self.conv2(x)))
        # Transpose back to (batch, seq_len, out_channels)
        return x.transpose(1, 2)


class AttentionLayer(nn.Module):
    """
    Trainable Additive Attention Mechanism.
    Computes dynamic alignment scores across timesteps and produces
    both the weighted context vector and interpretable attention weights.
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, rnn_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # rnn_outputs: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.proj(rnn_outputs))  # (batch, seq_len, attention_dim)
        scores = self.v(scores).squeeze(-1)           # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)           # (batch, seq_len)

        # Context vector = weighted sum over time: sum_t(alpha_t * h_t)
        # (batch, 1, seq_len) x (batch, seq_len, hidden_dim) -> (batch, 1, hidden_dim)
        context = torch.bmm(weights.unsqueeze(1), rnn_outputs).squeeze(1)

        return context, weights


class DenseClassifier(nn.Module):
    """Classification head mapping representation vector to scalar logit."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = DENSE_UNITS,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# =====================================================================
# 1. Proposed Hybrid Model: 1D-CNN + BiLSTM + Attention + Dense
# =====================================================================
class RansomwareHybridModel(nn.Module):
    """
    Proposed Architecture:
      1D-CNN (Local Features)
         ↓
      BiLSTM (Temporal Dependencies)
         ↓
      Attention (Dynamic Timestep Focus)
         ↓
      Dense Classifier (Logits Output)
    """

    def __init__(
        self,
        in_features: int = N_FEATURES,
        cnn_filters: int = CNN_FILTERS,
        lstm_hidden: int = LSTM_HIDDEN_SIZE,
        lstm_layers: int = LSTM_LAYERS,
        dense_units: int = DENSE_UNITS,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.cnn = Conv1DFeatureExtractor(
            in_channels=in_features,
            out_channels=cnn_filters,
            kernel_size=CNN_KERNEL_SIZE,
            dropout=0.2,
        )

        self.bilstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=LSTM_DROPOUT if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * 2  # Bidirectional doubles the representation
        self.attention = AttentionLayer(hidden_dim=lstm_out_dim, attention_dim=64)
        self.classifier = DenseClassifier(in_dim=lstm_out_dim, hidden_dim=dense_units, dropout=dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (batch, seq_len, in_features)
        cnn_out = self.cnn(x)                         # (batch, seq_len, 64)
        lstm_out, _ = self.bilstm(cnn_out)             # (batch, seq_len, 128)
        context, attn_weights = self.attention(lstm_out)  # (batch, 128), (batch, seq_len)
        logits = self.classifier(context)              # (batch, 1)

        if return_attention:
            return logits, attn_weights
        return logits, None


# =====================================================================
# 2. Baseline Architecture 1: CNN Only
# =====================================================================
class CNNOnlyModel(nn.Module):
    """Ablation/Baseline A: 1D-CNN + Global Average Pooling + Classifier."""

    def __init__(
        self,
        in_features: int = N_FEATURES,
        cnn_filters: int = CNN_FILTERS,
        dense_units: int = DENSE_UNITS,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.cnn = Conv1DFeatureExtractor(
            in_channels=in_features,
            out_channels=cnn_filters,
            kernel_size=CNN_KERNEL_SIZE,
            dropout=0.2,
        )
        self.classifier = DenseClassifier(in_dim=cnn_filters, hidden_dim=dense_units, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cnn_out = self.cnn(x)                  # (batch, seq_len, 64)
        pooled = torch.mean(cnn_out, dim=1)    # Global average pooling across time -> (batch, 64)
        logits = self.classifier(pooled)
        return logits, None


# =====================================================================
# 3. Baseline Architecture 2: BiLSTM Only
# =====================================================================
class BiLSTMOnlyModel(nn.Module):
    """Ablation/Baseline B: BiLSTM + Global Pooling + Classifier."""

    def __init__(
        self,
        in_features: int = N_FEATURES,
        lstm_hidden: int = LSTM_HIDDEN_SIZE,
        lstm_layers: int = LSTM_LAYERS,
        dense_units: int = DENSE_UNITS,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=in_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=LSTM_DROPOUT if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2
        self.classifier = DenseClassifier(in_dim=lstm_out_dim, hidden_dim=dense_units, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        lstm_out, _ = self.bilstm(x)           # (batch, seq_len, 128)
        pooled = torch.mean(lstm_out, dim=1)   # Mean pooling across time -> (batch, 128)
        logits = self.classifier(pooled)
        return logits, None


# =====================================================================
# 4. Baseline Architecture 3: CNN + BiLSTM (Without Attention)
# =====================================================================
class CNN_BiLSTMModel(nn.Module):
    """Ablation/Baseline C: CNN + BiLSTM (no attention mechanism)."""

    def __init__(
        self,
        in_features: int = N_FEATURES,
        cnn_filters: int = CNN_FILTERS,
        lstm_hidden: int = LSTM_HIDDEN_SIZE,
        lstm_layers: int = LSTM_LAYERS,
        dense_units: int = DENSE_UNITS,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.cnn = Conv1DFeatureExtractor(
            in_channels=in_features,
            out_channels=cnn_filters,
            kernel_size=CNN_KERNEL_SIZE,
            dropout=0.2,
        )
        self.bilstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=LSTM_DROPOUT if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2
        self.classifier = DenseClassifier(in_dim=lstm_out_dim, hidden_dim=dense_units, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cnn_out = self.cnn(x)
        lstm_out, _ = self.bilstm(cnn_out)
        pooled = torch.mean(lstm_out, dim=1)   # Mean pooling across time
        logits = self.classifier(pooled)
        return logits, None


def get_model_summary(model: nn.Module) -> str:
    """Return human-readable summary of parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{model.__class__.__name__} - Total Params: {total_params:,}, Trainable: {trainable_params:,}"
