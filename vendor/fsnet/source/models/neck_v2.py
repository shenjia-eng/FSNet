from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ff_parser import FFParserBlock
from .freq_spatial_gate import FreqSpatialGate
from .parallel_fusion import P5GlobalContext, SEChannelAttention


class CrossScaleFFParser(nn.Module):
    """Fuse P3/P4/P5 at P3 resolution, apply one FFParser, residual back to each scale."""

    def __init__(
        self,
        channels: int = 256,
        p3_ff_hw=None,
        use_freq_attn: bool = False,
        attn_reduction: int = 16,
    ):
        super().__init__()
        g = 32 if channels % 32 == 0 else 1
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.GroupNorm(g, channels),
            nn.ReLU(inplace=True),
        )
        fh, fw = (None, None)
        if p3_ff_hw is not None:
            fh, fw = int(p3_ff_hw[0]), int(p3_ff_hw[1])
        self.ff = FFParserBlock(
            channels,
            h=fh,
            w=fw,
            use_freq_attn=use_freq_attn,
            attn_reduction=attn_reduction,
        )
        self.gamma_p4 = nn.Parameter(torch.tensor(0.5))
        self.gamma_p5 = nn.Parameter(torch.tensor(0.5))

    def forward(self, p3, p4, p5):
        h3, w3 = p3.shape[-2], p3.shape[-1]
        p4a = F.interpolate(p4, size=(h3, w3), mode="nearest")
        p5a = F.interpolate(p5, size=(h3, w3), mode="nearest")
        shared = self.ff(self.fuse(torch.cat([p3, p4a, p5a], dim=1)))
        p3_out = p3 + shared
        p4_delta = F.interpolate(shared, size=p4.shape[-2:], mode="nearest")
        p5_delta = F.interpolate(shared, size=p5.shape[-2:], mode="nearest")
        p4_out = p4 + self.gamma_p4 * p4_delta
        p5_out = p5 + self.gamma_p5 * p5_delta
        return p3_out, p4_out, p5_out


class NeckV2(nn.Module):
    """Single-path detection neck after cross-scale FF."""

    def __init__(
        self,
        channels: int = 256,
        use_high_res_fpn: bool = True,
        det_ff_hw=None,
        use_freq_spatial_gate: bool = True,
        use_freq_attn: bool = False,
        attn_reduction: int = 16,
    ):
        super().__init__()
        self.use_high_res_fpn = use_high_res_fpn
        self.p5_context = P5GlobalContext(channels)
        self.se = SEChannelAttention(channels)
        fh, fw = (None, None)
        if det_ff_hw is not None:
            fh, fw = int(det_ff_hw[0]), int(det_ff_hw[1])
        self.ffparser_det = FFParserBlock(
            channels,
            h=fh,
            w=fw,
            use_freq_attn=use_freq_attn,
            attn_reduction=attn_reduction,
        )
        self.freq_spatial_gate = (
            FreqSpatialGate(channels=channels) if use_freq_spatial_gate else None
        )

    def forward(self, p3, p4, p5, fpn_fuse):
        ctx = self.p5_context(p5, p3.shape[-2:])
        if self.use_high_res_fpn and fpn_fuse is not None:
            p4_up = p4
            if p4.shape[-2:] != p3.shape[-2:]:
                p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
            det = fpn_fuse(torch.cat([p3, p4_up], dim=1)) + ctx
        else:
            det = p4 + ctx
        det = self.se(det)
        f_ff = self.ffparser_det(det)
        if self.freq_spatial_gate is not None:
            p3_ref = p3
            if p3_ref.shape[-2:] != det.shape[-2:]:
                p3_ref = F.interpolate(
                    p3_ref, size=det.shape[-2:], mode="bilinear", align_corners=False
                )
            det = self.freq_spatial_gate(f_ff, p3_ref)
        else:
            det = f_ff
        return det
