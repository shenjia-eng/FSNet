# Spatial gating adds adaptive feature modulation to FSNet.

import torch
import torch.nn as nn


class FreqSpatialGate(nn.Module):
    """Spatial signal modulates the main det path (multiply or additive residual).

    mul:  f_out = f_main * (1 + alpha * sigmoid(conv_gate(f_sd)))
    add:  f_out = f_main + alpha * conv_spatial(f_sd)
    """

    def __init__(self, channels=256, init_alpha=0.1, gate_bias_init=-4.0, mode='mul'):
        super().__init__()
        if mode not in ('mul', 'add'):
            raise ValueError("mode must be 'mul' or 'add', got %r" % mode)
        self.mode = mode
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        if mode == 'mul':
            self.spatial_proj = None
            self.conv_gate = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            )
            nn.init.constant_(self.conv_gate[-1].bias, float(gate_bias_init))
        else:
            self.conv_gate = None
            self.spatial_proj = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            )
            nn.init.zeros_(self.spatial_proj[-1].weight)
            nn.init.zeros_(self.spatial_proj[-1].bias)

    def forward(self, f_main, f_sd):
        if self.mode == 'add':
            out = f_main + self.alpha * self.spatial_proj(f_sd)
            self.last_gate = None
            return out
        gate = torch.sigmoid(self.conv_gate(f_sd))
        self.last_gate = gate.detach()
        return f_main * (1.0 + self.alpha * gate)
