"""Channel-group Mamba refine adapted from S3Mamba _process_channel_group."""
import warnings

import torch
import torch.nn as nn


class _ChannelSeqGRU(nn.Module):
    """Fallback when mamba_ssm is unavailable (e.g. Windows without CUDA ext)."""

    def __init__(self, d_model, expand=2):
        super().__init__()
        hidden = d_model * expand
        self.gru = nn.GRU(d_model, hidden, batch_first=True)
        self.out = nn.Linear(hidden, d_model)

    def forward(self, x):
        y, _ = self.gru(x)
        return self.out(y)


def _make_channel_seq_block(d_model):
    try:
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
        except ImportError:
            from mamba_ssm import Mamba

        return Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2), 'mamba'
    except ImportError:
        warnings.warn(
            'mamba_ssm not installed; ChannelMambaRefine uses bidirectional GRU fallback.',
            stacklevel=3,
        )
        return _ChannelSeqGRU(d_model), 'gru'


class ChannelMambaRefine(nn.Module):
    def __init__(self, in_channels=256, mid_channels=128, num_groups=32, group_norm_groups=4):
        super().__init__()
        if mid_channels % num_groups != 0:
            raise ValueError('mid_channels must be divisible by num_groups')
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_groups = num_groups
        self.d_model = mid_channels // num_groups

        self.pre_proj = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.mamba, backend_fwd = _make_channel_seq_block(self.d_model)
        self.mamba_flip, backend_bwd = _make_channel_seq_block(self.d_model)
        self.seq_backend = backend_fwd if backend_fwd == backend_bwd else '%s/%s' % (backend_fwd, backend_bwd)
        self.proj = nn.Sequential(nn.GroupNorm(group_norm_groups, mid_channels), nn.SiLU())
        self.proj_flip = nn.Sequential(nn.GroupNorm(group_norm_groups, mid_channels), nn.SiLU())
        self.post_proj = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=True)

    def _run_mamba_pair(self, x):
        b, c, h, w = x.size()
        flat = x.view(b * h * w, c)

        seq = flat.view(b * h * w, self.num_groups, self.d_model)
        fwd = self.mamba(seq).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        fwd = self.proj(fwd)

        flip = torch.flip(flat, dims=(-1,))
        flip_seq = flip.view(b * h * w, self.num_groups, self.d_model)
        bwd = self.mamba_flip(flip_seq).view(b, h, w, c).flip(dims=(-1,)).permute(0, 3, 1, 2).contiguous()
        bwd = self.proj_flip(bwd)

        return fwd + x, bwd + x

    def forward(self, x):
        residual = x
        x = self.pre_proj(x)
        fwd, bwd = self._run_mamba_pair(x)
        x = 0.5 * (fwd + bwd)
        x = self.post_proj(x)
        return residual + x
