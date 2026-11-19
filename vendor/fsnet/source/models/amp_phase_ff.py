"""
Amplitude-Phase Frequency Decomposition + High-Frequency Guided Fusion.

Reference: Oppenheim & Lim (1981) "The importance of phase in signals"
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 频率掩码工具
# ---------------------------------------------------------------------------

def _freq_distance(h, w, device=None):
    """每个频率点到原点的归一化距离 (0~1). h=空间高, w=rfft宽=W//2+1."""
    fy = torch.arange(h, device=device, dtype=torch.float32) / max(h, 1)
    fx = torch.arange(w, device=device, dtype=torch.float32) / max(w, 1)
    # 低频在(0,0)角，不是中心；对于 rfft 这就是正确的
    fy, fx = torch.meshgrid(fy, fx, indexing='ij')
    dist = torch.sqrt(fy ** 2 + fx ** 2)
    return dist / max(dist.max(), torch.tensor(1e-8, device=device))


def _hf_mask(h, w, cutoff=0.4, smooth=0.1, device=None):
    """可微分的软高频掩码：距离>cutoff → 接近1, 距离<cutoff-smooth → 接近0."""
    d = _freq_distance(h, w, device)
    return torch.sigmoid((d - cutoff) / max(smooth, 1e-8))


# ---------------------------------------------------------------------------
# 幅度-相位分解 FFT 模块
# ---------------------------------------------------------------------------

class AmpPhaseFFT(nn.Module):
    """FFT 分解为幅度谱和相位谱，分别学习调制。

    - 幅度路径：per-channel per-frequency sigmoid gate -> 缩放到 [0.9, 1.1]
    - 相位路径：per-channel per-frequency additive bias -> 缩放到 ±π/8
    - 可选高频掩码：只在 HF 区域调制相位（保护低频全局结构）
    """

    def __init__(self, dim, h=None, w=None, hf_phase_only=True, cutoff=0.4):
        super().__init__()
        self.dim = dim
        self.hf_phase_only = hf_phase_only
        self.cutoff = cutoff
        if h is not None and w is not None:
            self._init_weights(h, w)

    def _init_weights_if_needed(self, fh, fw, device):
        if not hasattr(self, 'amp_logit'):
            self._init_weights(fh, fw)
            self.to(device)

    def _init_weights(self, fh, fw):
        # 幅度门控：初始化为 0 → sigmoid(0)=0.5 → 不改变幅度
        self.amp_logit = nn.Parameter(torch.zeros(self.dim, fh, fw))
        # 相位偏置：初始化为 0 → 不改变相位
        self.phase_bias = nn.Parameter(torch.zeros(self.dim, fh, fw))
        # 高频掩码（固定）用于相位保护
        self.register_buffer('_hf_mask', _hf_mask(fh, fw, self.cutoff, device=None), persistent=False)

    def forward(self, x):
        B, C, H, W = x.shape
        if C != self.dim:
            raise ValueError(f'AmpPhaseFFT expects {self.dim} channels, got {C}')
        fh, fw = H, W // 2 + 1
        self._init_weights_if_needed(fh, fw, x.device)

        dtype = x.dtype
        x = x.float()

        # ---- FFT 分解 ----
        X_freq = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        amp = torch.abs(X_freq)                    # (B,C,fh,fw)  幅度谱
        phase = torch.angle(X_freq)                # (B,C,fh,fw)  相位谱

        amp_gate = self.amp_logit.sigmoid()        # (C,fh,fw) ∈ [0,1]
        # 插值到当前尺寸
        if amp_gate.shape[-2:] != (fh, fw):
            amp_gate = F.interpolate(
                amp_gate.unsqueeze(0).float(), size=(fh, fw), mode='bilinear', align_corners=False
            ).squeeze(0).to(amp.dtype)
        amp = amp * (0.9 + 0.2 * amp_gate)           # 范围 [0.9, 1.1]×原始幅度

        # ---- 相位调制（高频聚焦）----
        phase_shift = self.phase_bias.tanh() * (math.pi / 8)  # (C,fh,fw) ∈ [-π/8, π/8]
        if phase_shift.shape[-2:] != (fh, fw):
            phase_shift = F.interpolate(
                phase_shift.unsqueeze(0).float(), size=(fh, fw), mode='bilinear', align_corners=False
            ).squeeze(0).to(phase.dtype)
        hf_mask = self._hf_mask.to(device=x.device, dtype=phase.dtype)
        if hf_mask.shape != (fh, fw):
            hf_mask = F.interpolate(
                hf_mask.unsqueeze(0).unsqueeze(0), size=(fh, fw), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
        if self.hf_phase_only:
            phase = phase + phase_shift * hf_mask   # 仅高频相位被调制
        else:
            phase = phase + phase_shift

        # ---- 重建 ----
        X_new = amp * torch.exp(1j * phase)
        x_out = torch.fft.irfft2(X_new, s=(H, W), dim=(2, 3), norm='ortho')
        return x_out.to(dtype)


class AmpPhaseFFTBlock(nn.Module):
    """Residual wrapper: x + gamma * AmpPhaseFFT(x), gamma starts at 0."""
    def __init__(self, dim, h=None, w=None, hf_phase_only=True, cutoff=0.4):
        super().__init__()
        self.ff = AmpPhaseFFT(dim, h=h, w=w, hf_phase_only=hf_phase_only, cutoff=cutoff)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x + self.gamma * self.ff(x)


# ---------------------------------------------------------------------------
# 跨域高频融合（频域相位 HF + 空域边缘 HF）
# ---------------------------------------------------------------------------

class CrossDomainHFFusion(nn.Module):
    """融合频域高频相位特征和空域高通边缘特征。

    - 频域分支：AmpPhaseFFT 的相位高频输出 (结构/边界位置)
    - 空域分支：FPN P3 的高分辨率特征 (含 SpatialDiff 边缘)
    - 融合方式：通道门控 + 池化空间注意力（避免 O(HW²) 复杂度）
    """

    def __init__(self, channels=256, hf_cutoff=0.4, pool_size=16):
        super().__init__()
        self.pool_size = pool_size

        # 通道门控：频域特征通过 SE 门控空域特征
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )
        # 空域门控：空域特征通过 SE 门控频域特征
        self.spatial_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

        # 池化空间注意力：降采样后计算轻量交叉注意力
        self.q_proj = nn.Conv2d(channels, channels // 8, 1)
        self.k_proj = nn.Conv2d(channels, channels // 8, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

        # 输出融合
        self.out_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, spatial_feat, freq_feat):
        """
        Args:
            spatial_feat: 空域特征 (B,C,H,W) — FPN P3
            freq_feat:   频域处理后特征 (B,C,H',W') — AmpPhaseFFT (det_feat)
        Returns:
            fused: (B,C,H',W') 与 freq_feat 同尺寸
        """
        # 对齐空间尺寸
        if spatial_feat.shape[-2:] != freq_feat.shape[-2:]:
            spatial_feat = F.interpolate(
                spatial_feat, size=freq_feat.shape[-2:], mode='bilinear', align_corners=False
            )

        # 1) 通道门控互调
        s_gate = self.freq_gate(freq_feat)      # 频域 → 门控空域
        f_gate = self.spatial_gate(spatial_feat) # 空域 → 门控频域
        s = spatial_feat * s_gate
        f = freq_feat * f_gate

        # 2) 池化空间交叉注意力（固定到 pool_size × pool_size，O(p⁴) 可控）
        H, W = s.shape[-2:]
        ps = min(self.pool_size, H, W)
        s_pool = F.adaptive_avg_pool2d(s, ps)
        f_pool = F.adaptive_avg_pool2d(f, ps)

        B, C = s_pool.shape[:2]
        HW_p = ps * ps
        head_dim = max(C // 8, 8)

        Q = self.q_proj(s_pool).view(B, head_dim, HW_p).permute(0, 2, 1)  # (B, ps², hd)
        K = self.k_proj(f_pool).view(B, head_dim, HW_p)                    # (B, hd, ps²)
        V = self.v_proj(f_pool).view(B, C, HW_p).permute(0, 2, 1)         # (B, ps², C)

        attn = (Q @ K) / math.sqrt(head_dim)   # (B, ps², ps²)  e.g. 256×256 ✓
        attn = attn.softmax(dim=-1)
        attended_pool = (attn @ V).permute(0, 2, 1).view(B, C, ps, ps)

        # 上采样回原始尺寸
        attended = F.interpolate(attended_pool, size=(H, W), mode='bilinear', align_corners=False)

        # 3) 残差融合
        enhanced = self.out_fuse(torch.cat([spatial_feat, f + attended], dim=1))
        return enhanced


class HighFreqPhaseGuide(nn.Module):
    def __init__(self, channels=256, det_h=None, det_w=None,
                 hf_phase_only=True, cutoff=0.4):
        super().__init__()
        self.amp_phase_ff = AmpPhaseFFTBlock(
            channels, h=det_h, w=det_w,
            hf_phase_only=hf_phase_only, cutoff=cutoff,
        )
        self.cross_fuse = CrossDomainHFFusion(channels, cutoff)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, det_feat, spatial_hf=None):
        """
        Args:
            det_feat: 检测特征 (B,256,H,W)
            spatial_hf: 可选的空域高频特征，默认用 det_feat 自身
        """
        # 频域 amp/phase 处理
        freq_out = self.amp_phase_ff(det_feat)

        # 跨域融合（如果提供了空域高频特征）
        if spatial_hf is not None:
            freq_out = self.cross_fuse(spatial_hf, freq_out)

        return det_feat + self.gamma * freq_out
