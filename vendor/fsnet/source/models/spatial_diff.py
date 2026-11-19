"""SDS brings learnable spatial structure features into FSNet."""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


def _discrete_laplacian_kernel(kernel_size: int) -> torch.Tensor:
    """Standard discrete Laplacian of given size: center = N_neighbors, each neighbor = -1.

    For 3x3: center=8, 8-neighbor = -1 each, sum=0.
    For 7x7: center=48, 48-neighbor = -1 each, sum=0.
    """
    k = torch.zeros(kernel_size, kernel_size)
    h = kernel_size // 2
    k[h, h] = float(kernel_size * kernel_size - 1)
    k[k != k[h, h]] = -1.0
    return k



def _gaussian_kernel2d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    g = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / (g.sum() + 1e-8)


def _gaussian_derivative_kernels(size: int, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    g = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    gx = -xx / (sigma ** 2) * g
    gy = -yy / (sigma ** 2) * g
    gx = gx / (gx.abs().sum() + 1e-8)
    gy = gy / (gy.abs().sum() + 1e-8)
    return gx, gy


def _scharr_derivative_kernels(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed 3x3 Scharr in the center of a size x size kernel."""
    gx3 = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]])
    gy3 = gx3.t()
    gx = torch.zeros(size, size)
    gy = torch.zeros(size, size)
    off = (size - 3) // 2
    gx[off : off + 3, off : off + 3] = gx3
    gy[off : off + 3, off : off + 3] = gy3
    gx = gx / (gx.abs().sum() + 1e-8)
    gy = gy / (gy.abs().sum() + 1e-8)
    return gx, gy


def _init_conv_from_kernel(conv: nn.Conv2d, kernel2d: torch.Tensor) -> None:
    with torch.no_grad():
        w = conv.weight
        for oc in range(w.shape[0]):
            for ic in range(w.shape[1]):
                w[oc, ic].copy_(kernel2d)


class SDifferenceConv2d(nn.Module):
    """Laplacian-style learnable high-pass: center = sum(weights), neighbors negated."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        grad_weight = -self.weight.clone()
        h, w = grad_weight.shape[-2:]
        grad_weight[..., h // 2, w // 2] = torch.sum(self.weight, dim=(-2, -1))
        return F.conv2d(
            input,
            grad_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class FixedLaplacianConv2d(nn.Module):
    """Standard discrete-Laplacian centre-surround conv with FROZEN kernel.

    Weight is initialised from _discrete_laplacian_kernel(kernel_size) and
    never updated (requires_grad=False). Bias is optional and trainable.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        k2d = _discrete_laplacian_kernel(kernel_size[0])
        weight = k2d.expand(out_channels, in_channels, -1, -1).clone()
        self.register_buffer('weight', weight)
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            dilation=1,
            groups=1,
        )


class GradientMagnitudeConv2d(nn.Module):
    """Learnable Sobel/Scharr-style operator: sqrt(Gx^2 + Gy^2 + eps)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=7,
        padding=3,
        operator='sobel',
        bias=True,
    ):
        super().__init__()
        self.conv_x = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.conv_y = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        if operator == 'scharr':
            gx, gy = _scharr_derivative_kernels(kernel_size)
        else:
            gx, gy = _gaussian_derivative_kernels(kernel_size, sigma=1.0)
        _init_conv_from_kernel(self.conv_x, gx)
        _init_conv_from_kernel(self.conv_y, gy)

    def forward(self, x):
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        if self.bias is not None:
            mag = mag + self.bias.view(1, -1, 1, 1)
        return mag


class DoGConv2d(nn.Module):
    """Learnable difference-of-Gaussians band-pass (small sigma - large sigma)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=7,
        padding=3,
        sigma1=1.0,
        sigma2=2.0,
        bias=True,
    ):
        super().__init__()
        self.conv_small = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.conv_large = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        g1 = _gaussian_kernel2d(kernel_size, sigma1)
        g2 = _gaussian_kernel2d(kernel_size, sigma2)
        _init_conv_from_kernel(self.conv_small, g1)
        _init_conv_from_kernel(self.conv_large, g2)

    def forward(self, x):
        out = self.conv_small(x) - self.conv_large(x)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class LoGConv2d(nn.Module):
    """Gaussian smooth (depthwise) then Laplacian high-pass."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=7,
        padding=3,
        sigma=1.5,
        bias=True,
    ):
        super().__init__()
        self.smooth = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.laplacian = SDifferenceConv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias
        )
        g = _gaussian_kernel2d(kernel_size, sigma)
        with torch.no_grad():
            for ic in range(in_channels):
                self.smooth.weight[ic, 0].copy_(g)

    def forward(self, x):
        return self.laplacian(self.smooth(x))


STEM_TYPES = ('laplacian', 'plain', 'sobel', 'scharr', 'dog', 'log', 'fixed_laplacian')


class SpatialDiffStem(nn.Module):
    """Input 2D high-pass stem with residual.

    Args:
        in_channels: input image channels (3 for RGB)
        mid_channels: intermediate channel count
        kernel_size: conv kernel size (3 or 7)
        stem_type: 'laplacian' (learnable) | 'plain' | 'sobel' | 'scharr' | 'dog' | 'log' |
                   'fixed_laplacian' (frozen discrete Laplacian)
        freeze_diff: if True, freeze diff.* weights (for fixed_laplacian this is always True)
    """

    def __init__(self, in_channels=3, mid_channels=32, kernel_size=7,
                 stem_type='laplacian', freeze_diff=False):
        pad = kernel_size // 2
        super().__init__()
        self.stem_type = stem_type
        self.kernel_size = kernel_size

        if stem_type == 'plain':
            self.diff = nn.Conv2d(
                in_channels, mid_channels, kernel_size=kernel_size, padding=pad, bias=True,
            )
        elif stem_type == 'laplacian':
            self.diff = SDifferenceConv2d(
                in_channels, mid_channels, kernel_size=kernel_size, padding=pad
            )
        elif stem_type == 'fixed_laplacian':
            self.diff = FixedLaplacianConv2d(
                in_channels, mid_channels, kernel_size=kernel_size, padding=pad, bias=True
            )
            freeze_diff = True  # fixed_laplacian is always frozen
        elif stem_type == 'sobel':
            self.diff = GradientMagnitudeConv2d(
                in_channels, mid_channels, kernel_size=kernel_size, padding=pad, operator='sobel'
            )
        elif stem_type == 'scharr':
            self.diff = GradientMagnitudeConv2d(
                in_channels, mid_channels, kernel_size=kernel_size, padding=pad, operator='scharr'
            )
        elif stem_type == 'dog':
            self.diff = DoGConv2d(
                in_channels, mid_channels, kernel_size=kernel_size, padding=pad
            )
        elif stem_type == 'log':
            self.diff = LoGConv2d(
                in_channels, mid_channels, kernel_size=kernel_size, padding=pad
            )
        else:
            raise ValueError(
                f"stem_type must be one of {STEM_TYPES}, got {stem_type!r}"
            )

        if freeze_diff:
            for p in self.diff.parameters():
                p.requires_grad = False

        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(mid_channels, in_channels, kernel_size=1)

    def forward(self, x):
        return x + self.proj(self.act(self.bn(self.diff(x))))


class SpatialDiffFeature(nn.Module):
    """Feature-map high-pass branch for single-forward C3/C4/C5 fusion."""

    def __init__(self, channels, kernel_size=3):
        pad = kernel_size // 2
        super().__init__()
        groups = 32 if channels % 32 == 0 else 1
        self.diff = SDifferenceConv2d(
            channels, channels, kernel_size=kernel_size, padding=pad
        )
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.act(self.norm(self.diff(x)))
