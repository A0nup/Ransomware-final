"""
Advanced Shannon Entropy & Sliding Block Analysis Module
Calculates granular byte randomness distributions across file segments
to detect packed executables, hidden encrypted payloads, and ransomware artifacts.
"""

import math
from typing import Dict, Any, List, Tuple
import numpy as np
import plotly.graph_objects as go

from src.static_scanner import calculate_entropy


def compute_sliding_block_entropy(
    data: bytes,
    block_size: int = 512,
    step_size: int = 256,
) -> Dict[str, Any]:
    """
    Compute sliding window Shannon entropy across consecutive byte chunks of a file.
    Ransomware payloads and encrypted sections exhibit block entropy > 7.2.
    """
    total_len = len(data)
    if total_len == 0:
        return {
            "overall_entropy": 0.0,
            "max_block_entropy": 0.0,
            "min_block_entropy": 0.0,
            "avg_block_entropy": 0.0,
            "high_entropy_ratio": 0.0,
            "packing_verdict": "EMPTY_FILE",
            "offsets": [],
            "entropies": [],
        }

    overall_entropy = calculate_entropy(data)

    # Adjust block/step size for very small or large files
    if total_len < 1024:
        block_size = max(64, total_len // 4)
        step_size = max(32, block_size // 2)
    elif total_len > 10 * 1024 * 1024:
        block_size = 4096
        step_size = 2048

    offsets: List[int] = []
    entropies: List[float] = []

    pos = 0
    while pos < total_len:
        chunk = data[pos : pos + block_size]
        if not chunk:
            break
        ent = calculate_entropy(chunk)
        offsets.append(pos)
        entropies.append(round(ent, 3))
        pos += step_size

    high_entropy_blocks = sum(1 for e in entropies if e >= 7.2)
    total_blocks = max(1, len(entropies))
    high_entropy_ratio = high_entropy_blocks / total_blocks

    if high_entropy_ratio >= 0.50 or overall_entropy >= 7.2:
        packing_verdict = "HEAVILY_ENCRYPTED_OR_PACKED"
    elif high_entropy_ratio >= 0.15 or overall_entropy >= 6.8:
        packing_verdict = "SUSPICIOUS_HIGH_ENTROPY_SECTIONS"
    else:
        packing_verdict = "STANDARD_UNCOMPRESSED_DATA"

    return {
        "overall_entropy": round(overall_entropy, 3),
        "max_block_entropy": round(max(entropies), 3) if entropies else 0.0,
        "min_block_entropy": round(min(entropies), 3) if entropies else 0.0,
        "avg_block_entropy": round(float(np.mean(entropies)), 3) if entropies else 0.0,
        "high_entropy_ratio": round(high_entropy_ratio * 100.0, 1),
        "packing_verdict": packing_verdict,
        "offsets": offsets,
        "entropies": entropies,
        "total_blocks_analyzed": total_blocks,
    }


def create_entropy_profile_chart(entropy_result: Dict[str, Any], file_name: str = "Target File") -> go.Figure:
    """
    Generate an interactive Plotly visualization of block entropy distribution.
    """
    offsets = entropy_result["offsets"]
    entropies = entropy_result["entropies"]

    fig = go.Figure()

    # Base curve
    fig.add_trace(go.Scatter(
        x=offsets,
        y=entropies,
        mode="lines",
        name="Block Shannon Entropy",
        line=dict(color="#38bdf8", width=2),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.15)",
        hovertemplate="Byte Offset: %{x:,}<br>Entropy: %{y:.2f} / 8.0<extra></extra>",
    ))

    # Threshold for packed/encrypted ciphertext
    fig.add_hline(
        y=7.2,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Encrypted / Packed Threshold (H >= 7.2)",
        annotation_position="bottom right",
    )

    # Threshold for normal executable code
    fig.add_hline(
        y=6.0,
        line_dash="dot",
        line_color="#10b981",
        annotation_text="Typical Code Threshold (H ~ 6.0)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=f"Sliding Block Entropy Distribution ({file_name}) — Overall: {entropy_result['overall_entropy']}/8.0",
        xaxis_title="File Byte Offset",
        yaxis_title="Shannon Entropy (bits/byte)",
        yaxis=dict(range=[0, 8.2]),
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig
