"""Small ST-GCN for per-clip pose sequence classification.

Input shape: (B, C=2, T, V=23)
Output: (B, 34) logits for all 17 EVGS items × both limbs.
Trained at clip level; predictions averaged per (patient, side) at inference.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg


# Skeleton edges for body+feet 23-keypoint subset (COCO-WholeBody first 23).
# Body 0-16 + feet 17-22.
EDGES: list[tuple[int, int]] = [
    # face/head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # shoulders + arms
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    # torso
    (5, 11), (6, 12), (11, 12),
    # legs
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    # feet
    (15, 17), (15, 18), (15, 19),
    (16, 20), (16, 21), (16, 22),
]
NUM_NODES = 23


def build_adjacency() -> torch.Tensor:
    """Symmetric normalized adjacency with self-loops.  D^{-1/2} (A + I) D^{-1/2}"""
    A = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    for i, j in EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A += np.eye(NUM_NODES, dtype=np.float32)
    d = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.clip(d, 1.0, None))
    A_norm = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return torch.from_numpy(A_norm)


class STGCNBlock(nn.Module):
    """One spatial-temporal block: spatial GCN + temporal conv1d (with residual)."""

    def __init__(self, in_c: int, out_c: int, A: torch.Tensor, t_kernel: int = 9, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("A", A.clone())
        self.spatial = nn.Conv2d(in_c, out_c, kernel_size=1)
        pad = (t_kernel - 1) // 2
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=(t_kernel, 1), padding=(pad, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout),
        )
        if stride == 1 and in_c == out_c:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_c),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        res = self.residual(x)
        # spatial: y = einsum('vw, bctw -> bctv', A, spatial(x))
        y = self.spatial(x)
        y = torch.einsum("vw,bctw->bctv", self.A, y)
        y = self.temporal(y)
        return self.relu(y + res)


class STGCN(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 34, t_kernel: int = 9, dropout: float = 0.2):
        super().__init__()
        A = build_adjacency()
        self.register_buffer("A", A)

        self.data_bn = nn.BatchNorm1d(in_channels * NUM_NODES)
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 32, A, t_kernel, stride=1, dropout=dropout),
            STGCNBlock(32, 32, A, t_kernel, stride=1, dropout=dropout),
            STGCNBlock(32, 64, A, t_kernel, stride=2, dropout=dropout),
            STGCNBlock(64, 64, A, t_kernel, stride=1, dropout=dropout),
            STGCNBlock(64, 128, A, t_kernel, stride=2, dropout=dropout),
        ])
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        B, C, T, V = x.shape
        # input batch-norm
        x = x.permute(0, 1, 3, 2).reshape(B, C * V, T)
        x = self.data_bn(x)
        x = x.reshape(B, C, V, T).permute(0, 1, 3, 2)
        for blk in self.blocks:
            x = blk(x)
        # Global avg pool over (T, V)
        x = x.mean(dim=(2, 3))  # (B, 128)
        return self.head(x)
