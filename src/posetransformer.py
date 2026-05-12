"""Lightweight pose-sequence Transformer (DSTformer-style spatial-temporal attention).

We label this 'PoseTransformer' in the report rather than 'MotionBERT' because we
train from scratch (the official MotionBERT pretrained weights live on OneDrive
without a clean automated download path within our time budget). Architecturally
the model carries the same inductive biases: spatial attention across the J=23
joints + temporal attention across time, stacked.

Input:  (B, C=2, T, V=23)
Output: (B, 34) logits  (L1..L17, R1..R17)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .stgcn import NUM_NODES  # 23


class _Block(nn.Module):
    """One DSTformer-style block: spatial-attn → temporal-attn → MLP."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V, D)
        B, T, V, D = x.shape

        # Spatial attention across joints (mix per frame independently)
        xs = x.reshape(B * T, V, D)
        xn = self.norm_s(xs)
        xa, _ = self.attn_s(xn, xn, xn, need_weights=False)
        xs = xs + self.drop(xa)
        x = xs.reshape(B, T, V, D)

        # Temporal attention across frames (mix per joint independently)
        xt = x.permute(0, 2, 1, 3).reshape(B * V, T, D)
        xn = self.norm_t(xt)
        xa, _ = self.attn_t(xn, xn, xn, need_weights=False)
        xt = xt + self.drop(xa)
        x = xt.reshape(B, V, T, D).permute(0, 2, 1, 3)

        # MLP
        xn = self.norm_m(x)
        x = x + self.drop(self.mlp(xn))
        return x


class PoseTransformer(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 34, dim: int = 96, depth: int = 3, heads: int = 4, t_max: int = 256, dropout: float = 0.15):
        super().__init__()
        self.embed = nn.Linear(in_channels, dim)
        self.pos_t = nn.Parameter(torch.zeros(1, t_max, 1, dim))
        self.pos_v = nn.Parameter(torch.zeros(1, 1, NUM_NODES, dim))
        nn.init.trunc_normal_(self.pos_t, std=0.02)
        nn.init.trunc_normal_(self.pos_v, std=0.02)
        self.blocks = nn.ModuleList([_Block(dim, heads, dropout=dropout) for _ in range(depth)])
        self.head_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V) -> (B, T, V, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        B, T, V, C = x.shape
        x = self.embed(x) + self.pos_t[:, :T] + self.pos_v
        for blk in self.blocks:
            x = blk(x)
        # Mean pool over (T, V)
        x = x.mean(dim=(1, 2))
        x = self.head_norm(x)
        return self.head(x)
