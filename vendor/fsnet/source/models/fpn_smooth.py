"""Lightweight ResBlock smoothing for FPN outputs."""
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class FPNSmooth(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.smooth_p3 = ResBlock(channels, channels, stride=1)
        self.smooth_p4 = ResBlock(channels, channels, stride=1)
        self.smooth_p5 = ResBlock(channels, channels, stride=1)

    def forward(self, features_fpn):
        p3, p4, p5 = features_fpn
        return [self.smooth_p3(p3), self.smooth_p4(p4), self.smooth_p5(p5)]
