from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTopDownAfterFF(nn.Module):
    """P5_ff -> P4, P4_fused -> P3 via 1x1 + nearest upsample + residual add."""

    def __init__(self, channels: int = 256):
        super().__init__()
        g = 32 if channels % 32 == 0 else 1
        self.p5_to_p4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(g, channels),
            nn.ReLU(inplace=True),
        )
        self.p4_to_p3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(g, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, p3_ff, p4_ff, p5_ff):
        p5_ctx = self.p5_to_p4(p5_ff)
        if p5_ctx.shape[-2:] != p4_ff.shape[-2:]:
            p5_ctx = F.interpolate(
                p5_ctx, size=p4_ff.shape[-2:], mode="nearest"
            )
        p4_fused = p4_ff + p5_ctx

        p4_ctx = self.p4_to_p3(p4_fused)
        if p4_ctx.shape[-2:] != p3_ff.shape[-2:]:
            p4_ctx = F.interpolate(
                p4_ctx, size=p3_ff.shape[-2:], mode="nearest"
            )
        p3_fused = p3_ff + p4_ctx
        return p3_fused, p4_fused, p5_ff
