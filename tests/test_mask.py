"""Padding-mask convention check: True in a key_padding_mask means the
position is ignored. Attention over a padded batch must equal attention
over the unpadded sequence."""

from pathlib import Path
import sys

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils

TOLERANCE = 1e-6


def run() -> None:
    utils.set_seed(0)
    attention = nn.MultiheadAttention(embed_dim=64, num_heads=4,
                                      batch_first=True)
    attention.eval()
    query = torch.randn(1, 3, 64)
    keys_full = torch.randn(1, 7, 64)
    true_length = 5

    # Padded path: positions >= true_length are masked out (True = pad).
    mask = torch.zeros(1, 7, dtype=torch.bool)
    mask[0, true_length:] = True
    with torch.no_grad():
        padded, _ = attention(query, keys_full, keys_full,
                              key_padding_mask=mask)
        unpadded, _ = attention(query, keys_full[:, :true_length],
                                keys_full[:, :true_length])
    deviation = (padded - unpadded).abs().max().item()
    assert deviation <= TOLERANCE, f"mask deviation {deviation}"
