"""
Frequency filtering complements FSNet's spatial features.

Mechanisms (keyed by ffparser_strategy):
  complex       — GFNet-style per-channel × per-frequency complex static weight
  real_static   — GFNet-style real-valued static weight, no phase modulation
  dynamic       — DFFormer-style: static base filter × input-conditioned band modulation
  afno_mlp      — AFNO-inspired: per-frequency shared channel MLP in spectral domain
  no_param_fft  — FNet-style: IFFT(Re(FFT(X))), no learnable parameters

Weights are stored at training det_feat resolution; at forward time they are
interpolated if val/infer uses a different spatial size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Must match crowd_datasets/SHHA/SHHA.py random_crop half_h / half_w.
FSNET_PATCH_INPUT_SIDE = 128


def _freq_distance_grid(fh, fw, device, dtype=torch.float32):
    """Normalized radial distance in rFFT grid (0 at DC corner, ~1 at Nyquist)."""
    fy = torch.arange(fh, device=device, dtype=dtype) / max(fh, 1)
    fx = torch.arange(fw, device=device, dtype=dtype) / max(fw, 1)
    fy, fx = torch.meshgrid(fy, fx, indexing='ij')
    dist = torch.sqrt(fy ** 2 + fx ** 2)
    return dist / max(dist.max(), torch.tensor(1e-8, device=device, dtype=dtype))


def _hard_band_masks(num_bands, fh, fw, device, dtype=torch.float32):
    """Disjoint radial band masks for num_bands in {1, 2, 3}."""
    dist = _freq_distance_grid(fh, fw, device, dtype)
    if num_bands <= 1:
        return [torch.ones_like(dist)]
    if num_bands == 2:
        mid = 0.5
        low = (dist < mid).to(dtype)
        high = (dist >= mid).to(dtype)
        return [low, high]
    if num_bands == 3:
        b0 = (dist < 1.0 / 3.0).to(dtype)
        b1 = ((dist >= 1.0 / 3.0) & (dist < 2.0 / 3.0)).to(dtype)
        b2 = (dist >= 2.0 / 3.0).to(dtype)
        return [b0, b1, b2]
    raise ValueError(f'ffparser_num_bands must be 1, 2, or 3, got {num_bands}')


class FFParser(nn.Module):
    """GFNet-style per-channel × per-frequency complex static filter."""

    def __init__(self, dim, h=None, w=None, init_scale=0.02):
        super().__init__()
        self.dim = dim
        self.init_scale = init_scale
        if h is not None and w is not None:
            self.complex_weight = nn.Parameter(
                torch.randn(dim, h, w, 2, dtype=torch.float32) * init_scale
            )

    def _freq_hw(self, spatial_h, spatial_w):
        return spatial_h, spatial_w // 2 + 1

    def _init_weight_if_needed(self, fh, fw, device):
        if not hasattr(self, 'complex_weight'):
            self.complex_weight = nn.Parameter(
                torch.randn(self.dim, fh, fw, 2, device=device, dtype=torch.float32)
                * self.init_scale
            )

    def _weight_for_forward(self, fh, fw, device, dtype):
        self._init_weight_if_needed(fh, fw, device)
        weight = torch.view_as_complex(self.complex_weight)
        if weight.shape[1] != fh or weight.shape[2] != fw:
            real = F.interpolate(
                weight.real.unsqueeze(1),
                size=(fh, fw),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            imag = F.interpolate(
                weight.imag.unsqueeze(1),
                size=(fh, fw),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            weight = torch.complex(real, imag)
        return weight.to(device=device, dtype=dtype)

    def forward(self, x, return_filtered_amp=False):
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f'FFParser expects {self.dim} channels, got {c}')
        fh, fw = self._freq_hw(h, w)
        dtype = x.dtype
        x = x.float()
        x_freq = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = self._weight_for_forward(fh, fw, x_freq.device, x_freq.dtype)
        x_freq = x_freq * weight
        filtered_amp = None
        if return_filtered_amp:
            filtered_amp = torch.abs(x_freq)
        x = torch.fft.irfft2(x_freq, s=(h, w), dim=(2, 3), norm='ortho')
        x = x.to(dtype)
        if return_filtered_amp:
            return x, filtered_amp
        return x


class RealStaticFFParser(nn.Module):
    """Real-valued static frequency filter (no phase modulation)."""

    def __init__(self, dim, h=None, w=None, init_scale=0.02):
        super().__init__()
        self.dim = dim
        self.init_scale = init_scale
        if h is not None and w is not None:
            self.real_weight = nn.Parameter(
                torch.randn(dim, h, w, dtype=torch.float32) * init_scale
            )

    def _freq_hw(self, spatial_h, spatial_w):
        return spatial_h, spatial_w // 2 + 1

    def _init_weight_if_needed(self, fh, fw, device):
        if not hasattr(self, 'real_weight'):
            self.real_weight = nn.Parameter(
                torch.randn(self.dim, fh, fw, device=device, dtype=torch.float32)
                * self.init_scale
            )

    def _weight_for_forward(self, fh, fw, device, dtype):
        self._init_weight_if_needed(fh, fw, device)
        weight = self.real_weight
        if weight.shape[1] != fh or weight.shape[2] != fw:
            weight = F.interpolate(
                weight.unsqueeze(1),
                size=(fh, fw),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
        return weight.to(device=device, dtype=dtype)

    def forward(self, x, return_filtered_amp=False):
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f'RealStaticFFParser expects {self.dim} channels, got {c}')
        fh, fw = self._freq_hw(h, w)
        dtype = x.dtype
        x = x.float()
        x_freq = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        amp_weight = self._weight_for_forward(fh, fw, x_freq.device, x_freq.dtype)
        x_freq = x_freq * amp_weight
        filtered_amp = None
        if return_filtered_amp:
            filtered_amp = torch.abs(x_freq)
        x = torch.fft.irfft2(x_freq, s=(h, w), dim=(2, 3), norm='ortho')
        x = x.to(dtype)
        if return_filtered_amp:
            return x, filtered_amp
        return x


class DynamicBandFFParser(nn.Module):
    """DFFormer-style dynamic band modulation.

    K(X) = K₀ ⊙ γ(X) where:
      - K₀ is a static base complex filter (same as FFParser)
      - γ(X) is input-conditioned per-band modulation via MLP on pooled FFT amplitude
      - Uses 2 radial bands (low/high) to keep MLP compact
    """

    def __init__(self, dim, h=None, w=None, init_scale=0.02, num_bands=2, mlp_hidden=64):
        super().__init__()
        self.dim = dim
        self.num_bands = int(num_bands)
        self.init_scale = init_scale
        self.mlp_hidden = int(mlp_hidden)

        # Base static complex filter
        if h is not None and w is not None:
            self.base_weight = nn.Parameter(
                torch.randn(dim, h, w, 2, dtype=torch.float32) * init_scale
            )

        # MLP: pooled band energy → per-channel log-modulation
        # Input: dim * num_bands  (pooled band amplitude per channel)
        # Output: dim * num_bands (log-modulation per channel per band)
        in_dim = dim * self.num_bands
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, in_dim),
        )
        # Initialize to near-identity (γ ≈ 1 → no modulation)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def _freq_hw(self, spatial_h, spatial_w):
        return spatial_h, spatial_w // 2 + 1

    def _init_weight_if_needed(self, fh, fw, device):
        if not hasattr(self, 'base_weight'):
            self.base_weight = nn.Parameter(
                torch.randn(self.dim, fh, fw, 2, device=device, dtype=torch.float32)
                * self.init_scale
            )

    def _base_weight_complex(self, fh, fw, device, dtype):
        self._init_weight_if_needed(fh, fw, device)
        weight = torch.view_as_complex(self.base_weight)
        if weight.shape[1] != fh or weight.shape[2] != fw:
            real = F.interpolate(
                weight.real.unsqueeze(1),
                size=(fh, fw),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            imag = F.interpolate(
                weight.imag.unsqueeze(1),
                size=(fh, fw),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            weight = torch.complex(real, imag)
        return weight.to(device=device, dtype=dtype)

    def forward(self, x, return_filtered_amp=False):
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f'DynamicBandFFParser expects {self.dim} channels, got {c}')
        fh, fw = self._freq_hw(h, w)
        dtype = x.dtype
        x = x.float()
        x_freq = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')  # (B, C, fh, fw)

        # ---- Dynamic modulation ----
        amp = torch.abs(x_freq)  # (B, C, fh, fw)
        masks = _hard_band_masks(self.num_bands, fh, fw, amp.device, torch.float32)
        # Pool band energy per channel
        band_energy = []
        for mask in masks:
            # (B, C, fh, fw) * (1, fh, fw) → mean → (B, C)
            band_energy.append((amp * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3))
                               / (mask.sum() + 1e-8))
        # (B, num_bands*C)
        pooled = torch.cat(band_energy, dim=1)
        # MLP → log-space modulation → exp to get multiplicative factor
        log_gamma = self.mlp(pooled)  # (B, num_bands*C)
        gamma = torch.exp(log_gamma * 0.1)  # soft modulation via scaled exp
        gamma = gamma.view(b, self.num_bands, c)  # (B, num_bands, C)

        # ---- Apply per-band modulation to base filter ----
        base_w = self._base_weight_complex(fh, fw, x_freq.device, x_freq.dtype)  # (C, fh, fw)

        # Build modulated filter: sum over bands of gamma_band * mask_band * base_w
        modulated = torch.zeros_like(x_freq)
        for band_idx, mask in enumerate(masks):
            # gamma[:, band_idx, :, None, None] → (B, C, 1, 1)
            g = gamma[:, band_idx, :, None, None].to(dtype=modulated.dtype)
            m = mask.unsqueeze(0).unsqueeze(0).to(dtype=modulated.dtype)
            modulated = modulated + x_freq * (g * m * base_w.unsqueeze(0))

        filtered_amp = None
        if return_filtered_amp:
            filtered_amp = torch.abs(modulated)
        x = torch.fft.irfft2(modulated, s=(h, w), dim=(2, 3), norm='ortho')
        x = x.to(dtype)
        if return_filtered_amp:
            return x, filtered_amp
        return x


class AFNOSpectralMLP(nn.Module):
    """AFNO-inspired per-frequency shared channel MLP in the spectral domain.

    For each frequency position (u,v), apply a shared 2-layer MLP along channel dim:
      Y_{:,u,v} = X_{:,u,v} + W₂ σ(W₁ X_{:,u,v})
    where X_{:,u,v} is the (B, 2C) real+imag vector at that frequency.

    Uses mixed-precision-friendly: split complex into real/imag → MLP → recombine.
    """

    def __init__(self, dim, reduction=4, mlp_drop=0.0):
        super().__init__()
        self.dim = dim
        self.reduction = int(reduction)
        hidden = max(2 * dim // reduction, 32)

        # MLP operates on concatenated [real, imag] = 2*dim
        self.fc1 = nn.Linear(2 * dim, hidden, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(mlp_drop) if mlp_drop > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden, 2 * dim, bias=True)

        # Zero-init to start near identity
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _freq_hw(self, spatial_h, spatial_w):
        return spatial_h, spatial_w // 2 + 1

    def forward(self, x, return_filtered_amp=False):
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f'AFNOSpectralMLP expects {self.dim} channels, got {c}')
        fh, fw = self._freq_hw(h, w)
        dtype = x.dtype
        x = x.float()
        x_freq = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')  # (B, C, fh, fw) complex

        # Split complex → real+imag channels: (B, C, fh, fw) → (B, 2C, fh, fw)
        real = x_freq.real
        imag = x_freq.imag
        # Stack as (B, fh, fw, 2*C) for linear op
        z = torch.cat([real, imag], dim=1)  # (B, 2C, fh, fw)
        b2, c2, fh2, fw2 = z.shape

        # Shared MLP over frequency tokens: (B, 2C, fh, fw) → (B, fh*fw, 2C)
        z = z.permute(0, 2, 3, 1).reshape(b2, fh2 * fw2, c2)
        residual = z
        z = self.fc1(z)
        z = self.act(z)
        z = self.drop(z)
        z = self.fc2(z)
        z = z + residual  # residual connection

        # Reshape back: (B, fh*fw, 2C) → (B, fh, fw, 2C) → (B, 2C, fh, fw)
        z = z.reshape(b2, fh2, fw2, c2).permute(0, 3, 1, 2)

        # Split back into real/imag
        new_real = z[:, :c, :, :]
        new_imag = z[:, c:, :, :]
        x_freq = torch.complex(new_real, new_imag)

        filtered_amp = None
        if return_filtered_amp:
            filtered_amp = torch.abs(x_freq)
        x_out = torch.fft.irfft2(x_freq, s=(h, w), dim=(2, 3), norm='ortho')
        x_out = x_out.to(dtype)
        if return_filtered_amp:
            return x_out, filtered_amp
        return x_out


class NoParamFFT(nn.Module):
    """FNet-style no-parameter spectral transform.

    Y = IFFT(Re(FFT(X)))
    Discards the imaginary (phase) component of the FFT spectrum.
    Zero learnable parameters.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def _freq_hw(self, spatial_h, spatial_w):
        return spatial_h, spatial_w // 2 + 1

    def forward(self, x, return_filtered_amp=False):
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f'NoParamFFT expects {self.dim} channels, got {c}')
        dtype = x.dtype
        x = x.float()
        x_freq = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        x_freq_real = torch.complex(x_freq.real, torch.zeros_like(x_freq.real))
        filtered_amp = None
        if return_filtered_amp:
            filtered_amp = torch.abs(x_freq_real)
        x_out = torch.fft.irfft2(x_freq_real, s=(h, w), dim=(2, 3), norm='ortho')
        x_out = x_out.to(dtype)
        if return_filtered_amp:
            return x_out, filtered_amp
        return x_out


class IdentityFFParser(nn.Module):
    """Identity spectral branch."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, return_filtered_amp=False):
        if x.shape[1] != self.dim:
            raise ValueError(f'IdentityFFParser expects {self.dim} channels, got {x.shape[1]}')
        if return_filtered_amp:
            return x, None
        return x


# ---------------------------------------------------------------------------
# Additional frequency-filter variants
# ---------------------------------------------------------------------------

class MultiBandFFParser(nn.Module):
    """FFParser with 1/2/3 disjoint radial frequency bands."""

    def __init__(self, dim, h=None, w=None, num_bands=1, init_scale=0.02):
        super().__init__()
        if num_bands not in (1, 2, 3):
            raise ValueError(f'num_bands must be 1, 2, or 3, got {num_bands}')
        self.dim = dim
        self.num_bands = int(num_bands)
        self.init_scale = init_scale
        self.band_weights = nn.ParameterList()
        if h is not None and w is not None:
            fh, fw = h, w // 2 + 1
            for _ in range(self.num_bands):
                self.band_weights.append(
                    nn.Parameter(
                        torch.randn(dim, fh, fw, 2, dtype=torch.float32) * init_scale
                    )
                )

    def _freq_hw(self, spatial_h, spatial_w):
        return spatial_h, spatial_w // 2 + 1

    def _init_band_if_needed(self, fh, fw, device):
        while len(self.band_weights) < self.num_bands:
            self.band_weights.append(
                nn.Parameter(
                    torch.randn(self.dim, fh, fw, 2, device=device, dtype=torch.float32)
                    * self.init_scale
                )
            )

    def _weight_for_band(self, band_idx, fh, fw, device, dtype):
        self._init_band_if_needed(fh, fw, device)
        weight = torch.view_as_complex(self.band_weights[band_idx])
        if weight.shape[1] != fh or weight.shape[2] != fw:
            real = F.interpolate(
                weight.real.unsqueeze(1),
                size=(fh, fw),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            imag = F.interpolate(
                weight.imag.unsqueeze(1),
                size=(fh, fw),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            weight = torch.complex(real, imag)
        return weight.to(device=device, dtype=dtype)

    def forward(self, x, return_filtered_amp=False):
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f'MultiBandFFParser expects {self.dim} channels, got {c}')
        fh, fw = self._freq_hw(h, w)
        dtype = x.dtype
        x = x.float()
        x_freq = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        masks = _hard_band_masks(self.num_bands, fh, fw, x_freq.device, torch.float32)
        combined = torch.zeros_like(x_freq)
        for band_idx, mask in enumerate(masks):
            w_band = self._weight_for_band(band_idx, fh, fw, x_freq.device, x_freq.dtype)
            combined = combined + x_freq * w_band * mask
        filtered_amp = None
        if return_filtered_amp:
            filtered_amp = torch.abs(combined)
        x = torch.fft.irfft2(combined, s=(h, w), dim=(2, 3), norm='ortho')
        x = x.to(dtype)
        if return_filtered_amp:
            return x, filtered_amp
        return x


class ChannelSharedFFParser(nn.Module):
    """Frequency filter with a single complex weight shared across all channels."""

    def __init__(self, dim, h=None, w=None, init_scale=0.02):
        super().__init__()
        self.dim = dim
        self.init_scale = init_scale
        if h is not None and w is not None:
            fh, fw = h, w // 2 + 1
            self.shared_weight = nn.Parameter(
                torch.randn(1, fh, fw, 2, dtype=torch.float32) * init_scale
            )

    def _freq_hw(self, spatial_h, spatial_w):
        return spatial_h, spatial_w // 2 + 1

    def _init_weight_if_needed(self, fh, fw, device):
        if not hasattr(self, 'shared_weight'):
            self.shared_weight = nn.Parameter(
                torch.randn(1, fh, fw, 2, device=device, dtype=torch.float32)
                * self.init_scale
            )

    def _weight_for_forward(self, fh, fw, device, dtype):
        self._init_weight_if_needed(fh, fw, device)
        weight = torch.view_as_complex(self.shared_weight)
        if weight.shape[1] != fh or weight.shape[2] != fw:
            real = F.interpolate(
                weight.real.unsqueeze(1),
                size=(fh, fw),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            imag = F.interpolate(
                weight.imag.unsqueeze(1),
                size=(fh, fw),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            weight = torch.complex(real, imag)
        return weight.to(device=device, dtype=dtype)

    def forward(self, x, return_filtered_amp=False):
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f'ChannelSharedFFParser expects {self.dim} channels, got {c}')
        fh, fw = self._freq_hw(h, w)
        dtype = x.dtype
        x = x.float()
        x_freq = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = self._weight_for_forward(fh, fw, x_freq.device, x_freq.dtype)
        x_freq = x_freq * weight
        filtered_amp = None
        if return_filtered_amp:
            filtered_amp = torch.abs(x_freq)
        x = torch.fft.irfft2(x_freq, s=(h, w), dim=(2, 3), norm='ortho')
        x = x.to(dtype)
        if return_filtered_amp:
            return x, filtered_amp
        return x


class AmplitudeOnlyFFParser(nn.Module):
    """Frequency filter that learns only real-valued amplitude weights."""

    def __init__(self, dim, h=None, w=None, init_scale=0.02):
        super().__init__()
        self.dim = dim
        self.init_scale = init_scale
        if h is not None and w is not None:
            fh, fw = h, w // 2 + 1
            self.amp_weight = nn.Parameter(
                torch.randn(dim, fh, fw, dtype=torch.float32) * init_scale
            )

    def _freq_hw(self, spatial_h, spatial_w):
        return spatial_h, spatial_w // 2 + 1

    def _init_weight_if_needed(self, fh, fw, device):
        if not hasattr(self, 'amp_weight'):
            self.amp_weight = nn.Parameter(
                torch.randn(self.dim, fh, fw, device=device, dtype=torch.float32)
                * self.init_scale
            )

    def _weight_for_forward(self, fh, fw, device, dtype):
        self._init_weight_if_needed(fh, fw, device)
        weight = self.amp_weight
        if weight.shape[1] != fh or weight.shape[2] != fw:
            weight = F.interpolate(
                weight.unsqueeze(1),
                size=(fh, fw),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
        return F.softplus(weight).to(device=device, dtype=dtype)

    def forward(self, x, return_filtered_amp=False):
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f'AmplitudeOnlyFFParser expects {self.dim} channels, got {c}')
        fh, fw = self._freq_hw(h, w)
        dtype = x.dtype
        x = x.float()
        x_freq = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        amp = self._weight_for_forward(fh, fw, x_freq.device, x_freq.dtype)
        x_freq = x_freq * amp
        filtered_amp = None
        if return_filtered_amp:
            filtered_amp = torch.abs(x_freq)
        x = torch.fft.irfft2(x_freq, s=(h, w), dim=(2, 3), norm='ortho')
        x = x.to(dtype)
        if return_filtered_amp:
            return x, filtered_amp
        return x


# ---------------------------------------------------------------------------
# FFParserBlock — dispatcher for all strategy variants
# ---------------------------------------------------------------------------

class FFParserBlock(nn.Module):
    """Residual wrapper dispatcher: x + gamma * strategy(x), gamma starts at 0.

    Strategies:
      none           — identity
      complex        — GFNet-style complex static filter
      real_static    — real-valued static filter (no phase modulation)
      dynamic        — DFFormer-style dynamic band-modulated filter
      afno_mlp       — AFNO-inspired spectral channel MLP
      no_param_fft   — FNet-style no-parameter FFT transform
      multi_band     — MultiBandFFParser
      amplitude_only — AmplitudeOnlyFFParser
      channel_shared — ChannelSharedFFParser
    """

    _STRATEGY_MAP = {
        'none': 'IdentityFFParser',
        'complex': 'FFParser',
        'real_static': 'RealStaticFFParser',
        'dynamic': 'DynamicBandFFParser',
        'afno_mlp': 'AFNOSpectralMLP',
        'no_param_fft': 'NoParamFFT',
        'multi_band': 'MultiBandFFParser',
        'amplitude_only': 'AmplitudeOnlyFFParser',
        'channel_shared': 'ChannelSharedFFParser',
    }

    def __init__(self, dim, h=None, w=None, init_scale=0.02,
                 use_freq_attn=False, attn_reduction=16, num_bands=1,
                 ffparser_strategy='complex', dynamic_num_bands=2,
                 dynamic_mlp_hidden=64, afno_reduction=4):
        super().__init__()
        self.strategy = ffparser_strategy

        if ffparser_strategy == 'none':
            self.ff = IdentityFFParser(dim)
        elif ffparser_strategy == 'real_static':
            self.ff = RealStaticFFParser(dim, h=h, w=w, init_scale=init_scale)
        elif ffparser_strategy == 'dynamic':
            self.ff = DynamicBandFFParser(
                dim,
                h=h,
                w=w,
                init_scale=init_scale,
                num_bands=int(dynamic_num_bands),
                mlp_hidden=int(dynamic_mlp_hidden),
            )
        elif ffparser_strategy == 'afno_mlp':
            self.ff = AFNOSpectralMLP(dim, reduction=int(afno_reduction))
        elif ffparser_strategy == 'no_param_fft':
            self.ff = NoParamFFT(dim)
        elif ffparser_strategy == 'amplitude_only':
            self.ff = AmplitudeOnlyFFParser(dim, h=h, w=w, init_scale=init_scale)
        elif ffparser_strategy == 'channel_shared':
            self.ff = ChannelSharedFFParser(dim, h=h, w=w, init_scale=init_scale)
        elif ffparser_strategy == 'multi_band':
            self.ff = MultiBandFFParser(dim, h=h, w=w, num_bands=int(num_bands), init_scale=init_scale)
        elif int(num_bands) <= 1:
            self.ff = FFParser(dim, h=h, w=w, init_scale=init_scale)
        else:
            self.ff = MultiBandFFParser(
                dim, h=h, w=w, num_bands=int(num_bands), init_scale=init_scale,
            )
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        if self.strategy == 'none':
            return self.ff(x)
        return x + self.gamma * self.ff(x)


# ---------------------------------------------------------------------------
# Channel attention
# ---------------------------------------------------------------------------

class FreqAmpChannelAttention(nn.Module):
    """rFFT-amplitude channel attention (SFIEET DCT-SE analogue)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward_from_amp(self, x, filtered_amp):
        b, c = x.shape[:2]
        dtype = x.dtype
        amp_pooled = filtered_amp.mean(dim=(2, 3))
        attn = self.fc(amp_pooled).view(b, c, 1, 1).to(dtype=dtype)
        return x * attn

    def forward(self, x):
        b, c, h, w = x.shape
        x_freq = torch.fft.rfft2(x.float(), dim=(2, 3), norm='ortho')
        amp = torch.abs(x_freq)
        return self.forward_from_amp(x, amp)
