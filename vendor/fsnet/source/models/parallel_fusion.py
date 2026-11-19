"""Parallel fusion: sigmoid gated residual or softmax voting over branches."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def vote_delta(base, branch_out, weight):
    """Sigmoid path: base + weight * (branch - base)."""
    return base + weight * (branch_out - base)


class VoteLogit(nn.Module):
    """Scalar sigmoid gate; init near 0.12 (logit=-2) for warm-start training."""

    def __init__(self, init_logit=-2.0):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(float(init_logit)))

    def weight(self):
        return torch.sigmoid(self.logit)


class SoftmaxVote(nn.Module):
    """Softmax vote: out = sum_i softmax(logits)_i * branch_i; weights sum to 1."""

    def __init__(self, branch_names, init_other=-2.0):
        super().__init__()
        if not branch_names:
            raise ValueError('branch_names must be non-empty')
        self.branch_names = list(branch_names)
        logits = torch.full((len(self.branch_names),), float(init_other))
        logits[0] = 0.0
        self.logits = nn.Parameter(logits)

    def weights(self):
        return F.softmax(self.logits, dim=0)

    def fuse(self, branches):
        if len(branches) != len(self.branch_names):
            raise ValueError(
                'expected %d branches %s, got %d'
                % (len(self.branch_names), self.branch_names, len(branches))
            )
        aligned = [self._align_branch(b) for b in branches]
        w = self.weights()
        out = aligned[0] * w[0]
        for i in range(1, len(aligned)):
            out = out + aligned[i] * w[i]
        return out

    @staticmethod
    def _align_branch(x):
        return x

    def vote_dict(self):
        w = self.weights()
        return {name: float(w[i].item()) for i, name in enumerate(self.branch_names)}


class FusionBranchAlign(nn.Module):
    """1×1 + GroupNorm before cross-branch softmax fusion."""

    def __init__(self, channels, groups=32):
        super().__init__()
        g = groups if channels % groups == 0 else 1
        self.align = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(g, channels),
        )

    def forward(self, x):
        return self.align(x)


class AlignedSoftmaxVote(SoftmaxVote):
    """Softmax vote with optional per-branch 1×1+GN alignment."""

    def __init__(self, branch_names, channels, align_mask=None, init_other=-2.0):
        super().__init__(branch_names, init_other=init_other)
        align_mask = align_mask or [False] * len(branch_names)
        self.aligns = nn.ModuleList(
            [FusionBranchAlign(channels) if m else nn.Identity() for m in align_mask]
        )

    def fuse(self, branches):
        if len(branches) != len(self.branch_names):
            raise ValueError(
                'expected %d branches %s, got %d'
                % (len(self.branch_names), self.branch_names, len(branches))
            )
        aligned = [align(b) for align, b in zip(self.aligns, branches)]
        w = self.weights()
        out = aligned[0] * w[0]
        for i in range(1, len(aligned)):
            out = out + aligned[i] * w[i]
        return out


class ParallelStemVote(nn.Module):
    """Merge dual-backbone VGG outputs at C3/C4/C5."""

    def __init__(self, fusion_vote='softmax', init_logit=-2.0):
        super().__init__()
        self.fusion_vote = fusion_vote
        if fusion_vote == 'softmax':
            self.softmax_vote = SoftmaxVote(['raw', 'stem'], init_other=init_logit)
            self.w_stem = None
        elif fusion_vote == 'sigmoid':
            self.softmax_vote = None
            self.w_stem = VoteLogit(init_logit)
        else:
            raise ValueError("fusion_vote must be 'sigmoid' or 'softmax'")

    def merge_levels(self, feat_raw, feat_stem, levels=(1, 2, 3)):
        merged = list(feat_raw)
        for idx in levels:
            if self.fusion_vote == 'softmax':
                merged[idx] = self.softmax_vote.fuse([feat_raw[idx], feat_stem[idx]])
            else:
                merged[idx] = vote_delta(feat_raw[idx], feat_stem[idx], self.w_stem.weight())
        return merged

    def vote_dict(self):
        if self.fusion_vote == 'softmax':
            d = self.softmax_vote.vote_dict()
            return {'raw': d['raw'], 'stem': d['stem']}
        return {'stem': float(self.w_stem.weight().item())}


class ConcatBranchFuse(nn.Module):
    """cat(base, branch) -> 1x1 + GroupNorm + ReLU."""

    def __init__(self, channels=256, groups=32):
        super().__init__()
        g = groups if channels % groups == 0 else 1
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(g, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, base, branch):
        return self.fuse(torch.cat([base, branch], dim=1))


class FPNConcatFuseMultiScale(nn.Module):
    """Per-scale concatenation fusion with an FFParser branch."""

    def __init__(self, channels=256):
        super().__init__()
        self.scales = ('p3', 'p4', 'p5')
        self.fusers = nn.ModuleDict({
            s: ConcatBranchFuse(channels) for s in self.scales
        })

    def forward_scale(self, scale, base, branch_fn):
        return self.fusers[scale](base, branch_fn(base))


class DetDualFreqAttention(nn.Module):
    """Parallel GAP-SE and rFFT-amplitude SE with learnable beta."""

    def __init__(
        self,
        channels=256,
        reduction=16,
        beta_init=0.1,
        identity_init=True,
        identity_bias=3.0,
    ):
        super().__init__()
        from .ff_parser import FreqAmpChannelAttention

        self.gap_se = SEChannelAttention(
            channels, reduction, identity_init=identity_init, identity_bias=identity_bias,
        )
        self.rfft_se = FreqAmpChannelAttention(channels, reduction)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        gap = self.gap_se(x)
        rfft = self.rfft_se(x)
        b = self.beta
        return x + b * (gap - x) + b * (rfft - x)


class SEChannelAttention(nn.Module):
    """SE-style channel recalibration."""

    def __init__(self, channels=256, reduction=16,
                 identity_init=True, identity_bias=3.0):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
        if identity_init:
            nn.init.zeros_(self.fc[-2].weight)
            nn.init.constant_(self.fc[-2].bias, identity_bias)

    def forward(self, x):
        w = self.fc(self.pool(x).flatten(1)).view(x.size(0), x.size(1), 1, 1)
        self.last_weights = w.detach()
        return x * w


class P5GlobalContext(nn.Module):
    """pool(P5) -> MLP -> broadcast add to P3-resolution detection features."""

    def __init__(self, channels=256, pool_type='avg'):
        super().__init__()
        if pool_type not in ('avg', 'max'):
            raise ValueError(f"pool_type must be 'avg' or 'max', got {pool_type!r}")
        self.pool_type = pool_type
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )

    def _pool(self, p5):
        if self.pool_type == 'max':
            return F.adaptive_max_pool2d(p5, 1)
        return F.adaptive_avg_pool2d(p5, 1)

    def forward(self, p5, target_size):
        ctx = self.mlp(self._pool(p5).flatten(1))
        ctx = ctx.view(ctx.size(0), ctx.size(1), 1, 1)
        if target_size != p5.shape[-2:]:
            ctx = F.interpolate(ctx, size=target_size, mode='nearest')
        return ctx


class ParallelStemVotePerLevel(nn.Module):
    """
    Single VGG forward with SpatialDiffFeature at C3/C4.
    Per-level fusion with branch alignment on diff path.

    Supports three fusion modes:
      - 'softmax': sum-to-1 weights
      - 'sigmoid':  independent residual gates
      - 'concat':   concatenation + 1×1 fusion
    """

    VGG16_BN_LEVEL_CHANNELS = {1: 256, 2: 512}

    def __init__(self, fusion_vote='softmax', init_logit=-2.0, level_channels=None):
        super().__init__()
        self.fusion_vote = fusion_vote
        level_channels = level_channels or self.VGG16_BN_LEVEL_CHANNELS
        self.levels = sorted(level_channels.keys())
        if fusion_vote == 'concat':
            self.concat_fuse = nn.ModuleDict({
                str(k): nn.Sequential(
                    nn.Conv2d(level_channels[k] * 2, level_channels[k], kernel_size=1, bias=False),
                    nn.GroupNorm(32 if level_channels[k] % 32 == 0 else 1, level_channels[k]),
                    nn.ReLU(inplace=True),
                )
                for k in self.levels
            })
            self.votes = None
            self.w_diff = None
        elif fusion_vote == 'softmax':
            self.votes = nn.ModuleDict({
                str(k): AlignedSoftmaxVote(
                    ['raw', 'diff'],
                    level_channels[k],
                    align_mask=[False, True],
                    init_other=init_logit,
                )
                for k in self.levels
            })
            self.concat_fuse = None
            self.w_diff = None
        else:
            self.votes = None
            self.w_diff = nn.ModuleDict({
                str(k): VoteLogit(init_logit) for k in self.levels
            })
            self.align_diff = nn.ModuleDict({
                str(k): FusionBranchAlign(level_channels[k]) for k in self.levels
            })
            self.concat_fuse = None

    def merge_level(self, feat_raw, feat_diff, level_idx):
        key = str(level_idx)
        if self.fusion_vote == 'concat':
            return self.concat_fuse[key](torch.cat([feat_raw, feat_diff], dim=1))
        if self.fusion_vote == 'softmax':
            return self.votes[key].fuse([feat_raw, feat_diff])
        diff_a = self.align_diff[key](feat_diff)
        return vote_delta(feat_raw, diff_a, self.w_diff[key].weight())

    def vote_dict(self):
        out = {}
        if self.fusion_vote == 'concat':
            for k in self.levels:
                out['stem_c%s_concat' % k] = 1.0
            return out
        if self.fusion_vote == 'softmax':
            for k, vote in self.votes.items():
                for name, val in vote.vote_dict().items():
                    out['stem_c%s_%s' % (k, name)] = val
        else:
            for k, gate in self.w_diff.items():
                out['stem_c%s_diff' % k] = float(gate.weight().item())
        return out


class ParallelFPNVote(nn.Module):
    """Shared vote logits across P3/P4/P5."""

    def __init__(self, use_smooth=False, use_ffpn=False, fusion_vote='softmax', init_logit=-2.0):
        super().__init__()
        self.fusion_vote = fusion_vote
        self.use_smooth = use_smooth
        self.use_ffpn = use_ffpn
        names = ['base']
        if use_smooth:
            names.append('smooth')
        if use_ffpn:
            names.append('ffpn')
        if fusion_vote == 'softmax':
            self.softmax_vote = SoftmaxVote(names, init_other=init_logit)
            self.w_smooth = None
            self.w_ffpn = None
        else:
            self.softmax_vote = None
            self.w_smooth = VoteLogit(init_logit) if use_smooth else None
            self.w_ffpn = VoteLogit(init_logit) if use_ffpn else None

    def forward(self, base, smooth_fn=None, ff_fn=None):
        if self.fusion_vote == 'softmax':
            branches = [base]
            if self.use_smooth:
                branches.append(smooth_fn(base))
            if self.use_ffpn:
                branches.append(ff_fn(base))
            return self.softmax_vote.fuse(branches)

        out = base
        if self.w_smooth is not None and smooth_fn is not None:
            out = vote_delta(out, smooth_fn(base), self.w_smooth.weight())
        if self.w_ffpn is not None and ff_fn is not None:
            out = vote_delta(out, ff_fn(base), self.w_ffpn.weight())
        return out

    def vote_dict(self):
        if self.fusion_vote == 'softmax':
            return dict(self.softmax_vote.vote_dict())
        d = {'base': 1.0}
        if self.w_smooth is not None:
            d['smooth'] = float(self.w_smooth.weight().item())
        if self.w_ffpn is not None:
            d['ffpn'] = float(self.w_ffpn.weight().item())
        return d


class ParallelFPNVotePerScale(nn.Module):
    """Per-scale softmax vote with the FFParser branch aligned before fusion."""

    def __init__(self, use_ffpn=True, fusion_vote='softmax', init_logit=-2.0, channels=256):
        super().__init__()
        self.use_ffpn = use_ffpn
        self.fusion_vote = fusion_vote
        self.scales = ('p3', 'p4', 'p5')
        if not use_ffpn:
            return
        if fusion_vote == 'softmax':
            self.votes = nn.ModuleDict({
                s: AlignedSoftmaxVote(
                    ['base', 'ffpn'],
                    channels,
                    align_mask=[False, True],
                    init_other=init_logit,
                )
                for s in self.scales
            })
        else:
            self.votes = None
            self.w_ffpn = nn.ModuleDict({s: VoteLogit(init_logit) for s in self.scales})
            self.align_ff = nn.ModuleDict({s: FusionBranchAlign(channels) for s in self.scales})

    def forward_scale(self, scale, base, ff_fn=None):
        if not self.use_ffpn:
            return base
        if self.fusion_vote == 'softmax':
            return self.votes[scale].fuse([base, ff_fn(base)])
        ff_a = self.align_ff[scale](ff_fn(base))
        return vote_delta(base, ff_a, self.w_ffpn[scale].weight())

    def vote_dict(self):
        out = {}
        if not self.use_ffpn:
            return out
        if self.fusion_vote == 'softmax':
            for s, vote in self.votes.items():
                for name, val in vote.vote_dict().items():
                    out['fpn_%s_%s' % (s, name)] = val
        else:
            for s, gate in self.w_ffpn.items():
                out['fpn_%s_ffpn' % s] = float(gate.weight().item())
        return out


class ParallelDetTripleVote(nn.Module):
    """
    det_R = Fuse(P3_f, P4_f) + ctx
                det_F = Fuse(CA(P3_f), CA(P4_f)) + ctx
                ctx = broadcast(MLP(GAP(P5_f)))

    Supports three fusion modes:
      - 'softmax': sum-to-1 weights
      - 'sigmoid':  independent residual gates
      - 'concat':   concatenation + 1×1 fusion
    """

    def __init__(self, use_path_f=False, use_path_d=False, fusion_vote='softmax', init_logit=-2.0):
        super().__init__()
        self.fusion_vote = fusion_vote
        self.use_path_f = use_path_f
        self.use_path_d = use_path_d
        names = ['det_R']
        if use_path_f:
            names.append('det_F')
        if use_path_d:
            names.append('det_D')
        if fusion_vote == 'concat' and len(names) >= 2:
            in_ch = 256 * len(names)
            self.concat_fuse = nn.Sequential(
                nn.Conv2d(in_ch, 256, kernel_size=1, bias=False),
                nn.GroupNorm(32, 256),
                nn.ReLU(inplace=True),
            )
            self.softmax_vote = None
            self.w_F = None
            self.w_D = None
        elif fusion_vote == 'softmax':
            self.softmax_vote = SoftmaxVote(names, init_other=init_logit)
            self.w_F = None
            self.w_D = None
            self.concat_fuse = None
        else:
            self.softmax_vote = None
            self.w_F = VoteLogit(init_logit) if use_path_f else None
            self.w_D = VoteLogit(init_logit) if use_path_d else None
            self.concat_fuse = None

    def forward(self, det_r, det_f=None, det_d=None):
        if self.fusion_vote == 'concat' and self.concat_fuse is not None:
            parts = [det_r]
            if self.use_path_f and det_f is not None:
                parts.append(det_f)
            if self.use_path_d and det_d is not None:
                parts.append(det_d)
            if len(parts) >= 2:
                return self.concat_fuse(torch.cat(parts, dim=1))
            return det_r

        if self.fusion_vote == 'softmax':
            branches = [det_r]
            if self.use_path_f:
                branches.append(det_f)
            if self.use_path_d:
                branches.append(det_d)
            return self.softmax_vote.fuse(branches)

        out = det_r
        if self.w_F is not None and det_f is not None:
            out = vote_delta(out, det_f, self.w_F.weight())
        if self.w_D is not None and det_d is not None:
            out = vote_delta(out, det_d, self.w_D.weight())
        return out

    def vote_dict(self):
        if self.fusion_vote == 'concat':
            return {'det_concat': 1.0}
        if self.fusion_vote == 'softmax':
            return dict(self.softmax_vote.vote_dict())
        d = {'det_R': 1.0}
        if self.w_F is not None:
            d['det_F'] = float(self.w_F.weight().item())
        if self.w_D is not None:
            d['det_D'] = float(self.w_D.weight().item())
        return d


class ParallelDetRefineVote(nn.Module):
    """Fuse det_feat with optional FFParser / Wavelet / VSS branches.

    Supports three fusion modes:
      - 'softmax': sum-to-1 weights
      - 'sigmoid':  independent residual gates
      - 'concat':   concatenation + 1×1 fusion
    """

    def __init__(
        self,
        use_ffparser=False,
        use_channel_mamba=False,
        use_wavelet=False,
        use_vss=False,
        fusion_vote='softmax',
        init_logit=-2.0,
    ):
        super().__init__()
        self.fusion_vote = fusion_vote
        self.use_ffparser = use_ffparser
        self.use_channel_mamba = use_channel_mamba
        self.use_wavelet = use_wavelet
        self.use_vss = use_vss
        names = ['base']
        if use_ffparser:
            names.append('ffdet')
        if use_channel_mamba:
            names.append('mamba')
        if use_wavelet:
            names.append('wavelet')
        if use_vss:
            names.append('vss')
        self._names = names
        if fusion_vote == 'concat' and len(names) >= 2:
            in_ch = 256 * len(names)
            self.concat_fuse = nn.Sequential(
                nn.Conv2d(in_ch, 256, kernel_size=1, bias=False),
                nn.GroupNorm(32, 256),
                nn.ReLU(inplace=True),
            )
            self.softmax_vote = None
            self.w_ffdet = self.w_mamba = self.w_wave = self.w_vss = None
        elif fusion_vote == 'softmax':
            align = [False] + [True] * (len(names) - 1)
            self.softmax_vote = AlignedSoftmaxVote(names, 256, align_mask=align, init_other=init_logit)
            self.w_ffdet = None
            self.w_mamba = None
            self.w_wave = None
            self.w_vss = None
            self.concat_fuse = None
        else:
            self.softmax_vote = None
            self.w_ffdet = VoteLogit(init_logit) if use_ffparser else None
            self.w_mamba = VoteLogit(init_logit) if use_channel_mamba else None
            self.w_wave = VoteLogit(init_logit) if use_wavelet else None
            self.w_vss = VoteLogit(init_logit) if use_vss else None
            self.align_ffdet = FusionBranchAlign(256) if use_ffparser else None
            self.concat_fuse = None

    def forward(self, base, ff_fn=None, mamba_fn=None, wave_fn=None, vss_fn=None):
        if self.fusion_vote == 'concat' and self.concat_fuse is not None:
            parts = [base]
            if self.use_ffparser and ff_fn is not None:
                parts.append(ff_fn(base))
            if self.use_channel_mamba and mamba_fn is not None:
                parts.append(mamba_fn(base))
            if self.use_wavelet and wave_fn is not None:
                parts.append(wave_fn(base))
            if self.use_vss and vss_fn is not None:
                parts.append(vss_fn(base))
            if len(parts) >= 2:
                return self.concat_fuse(torch.cat(parts, dim=1))
            return base

        if self.fusion_vote == 'softmax':
            branches = [base]
            if self.use_ffparser:
                branches.append(ff_fn(base))
            if self.use_channel_mamba:
                branches.append(mamba_fn(base))
            if self.use_wavelet:
                branches.append(wave_fn(base))
            if self.use_vss:
                branches.append(vss_fn(base))
            return self.softmax_vote.fuse(branches)

        out = base
        if self.w_ffdet is not None and ff_fn is not None:
            out = vote_delta(out, self.align_ffdet(ff_fn(base)), self.w_ffdet.weight())
        if self.w_mamba is not None and mamba_fn is not None:
            out = vote_delta(out, mamba_fn(base), self.w_mamba.weight())
        if self.w_wave is not None and wave_fn is not None:
            out = vote_delta(out, wave_fn(base), self.w_wave.weight())
        if self.w_vss is not None and vss_fn is not None:
            out = vote_delta(out, vss_fn(base), self.w_vss.weight())
        return out

    def vote_dict(self):
        if self.fusion_vote == 'concat':
            return {'refine_concat': 1.0}
        if self.fusion_vote == 'softmax':
            d = dict(self.softmax_vote.vote_dict())
            return {('refine_' + k): v for k, v in d.items()}
        d = {'base': 1.0}
        if self.w_ffdet is not None:
            d['ffdet'] = float(self.w_ffdet.weight().item())
        if self.w_mamba is not None:
            d['mamba'] = float(self.w_mamba.weight().item())
        if self.w_wave is not None:
            d['wavelet'] = float(self.w_wave.weight().item())
        if self.w_vss is not None:
            d['vss'] = float(self.w_vss.weight().item())
        return d
