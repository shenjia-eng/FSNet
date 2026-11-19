"""
HRNet-W32 backbone adapter for FSNet.

Extracts multi-resolution features from HRNet stage4 output and formats them
to match FSNet's Decoder(256, 512, 512) interface.

HRNet stage4 branch channels (W32, Bottleneck ×4):
  branch[0]: stride-4,  128ch  → proj →  256ch (as C2/C3)
  branch[1]: stride-8,  256ch  → proj →  512ch (as C4)
  branch[2]: stride-16, 512ch  → proj →  512ch (as C5)

Returns: [none, C2_feat, C4_feat, C5_feat] — FSNet takes [1],[2],[3].
Anchor level: 2 (same as VGG — HRNet keeps stride-4 branch throughout).
"""
from __future__ import annotations

import os

import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add HRNet classification repo lib/ to path (override via HRNET_ROOT env)
_hrnet_repo = os.environ.get(
    "HRNET_ROOT",
    r"optional/HRNet-Image-Classification",
)
_hrnet_lib = os.path.join(_hrnet_repo, "lib")
_cls_hrnet_path = os.path.join(_hrnet_lib, "models", "cls_hrnet.py")
if not os.path.isfile(_cls_hrnet_path):
    raise FileNotFoundError(
        f"HRNet cls_hrnet not found at {_cls_hrnet_path}. Set HRNET_ROOT to the repo root."
    )
_spec = importlib.util.spec_from_file_location("_hrnet_cls_hrnet", _cls_hrnet_path)
_cls_hrnet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cls_hrnet)
HighResolutionNet = _cls_hrnet.HighResolutionNet
BN_MOMENTUM = _cls_hrnet.BN_MOMENTUM


def _build_hrnet_w32_config():
    """Build HRNet-W32 config dict matching cls_hrnet_w32 yaml."""
    return {
        "MODEL": {
            "EXTRA": {
                "STAGE1": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 1,
                    "BLOCK": "BOTTLENECK",
                    "NUM_BLOCKS": [4],
                    "NUM_CHANNELS": [64],
                    "FUSE_METHOD": "SUM",
                },
                "STAGE2": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 2,
                    "BLOCK": "BASIC",
                    "NUM_BLOCKS": [4, 4],
                    "NUM_CHANNELS": [32, 64],
                    "FUSE_METHOD": "SUM",
                },
                "STAGE3": {
                    "NUM_MODULES": 4,
                    "NUM_BRANCHES": 3,
                    "BLOCK": "BASIC",
                    "NUM_BLOCKS": [4, 4, 4],
                    "NUM_CHANNELS": [32, 64, 128],
                    "FUSE_METHOD": "SUM",
                },
                "STAGE4": {
                    "NUM_MODULES": 3,
                    "NUM_BRANCHES": 4,
                    "BLOCK": "BASIC",
                    "NUM_BLOCKS": [4, 4, 4, 4],
                    "NUM_CHANNELS": [32, 64, 128, 256],
                    "FUSE_METHOD": "SUM",
                },
            }
        }
    }


class HRNetBackbone(nn.Module):
    """HRNet-W32 feature extractor for FSNet — returns [none, C2, C3, C4]."""

    def __init__(self, pretrained_path: str = ""):
        super().__init__()
        cfg = _build_hrnet_w32_config()
        self.hrnet = HighResolutionNet(cfg)

        if pretrained_path and os.path.isfile(pretrained_path):
            self.hrnet.init_weights(pretrained_path)
            print(f"HRNet: loaded pretrained weights from {pretrained_path}")
        else:
            self.hrnet.init_weights()
            print(f"HRNet: initialized from scratch (pretrained path not found: {pretrained_path})")

        # Remove classification head
        del self.hrnet.incre_modules
        del self.hrnet.downsamp_modules
        del self.hrnet.final_layer
        del self.hrnet.classifier

        # proj0_fused: (32+64)=96ch → 128 → 256 (stride-4, as C2)
        # proj1: 64ch → 256 → 512 (stride-8, as C3)
        # proj2: 128ch → 256 → 512 (stride-16, as C4)
        self.proj0_fused = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.proj1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # HRNet stem
        x = self.hrnet.conv1(x)
        x = self.hrnet.bn1(x)
        x = self.hrnet.relu(x)
        x = self.hrnet.conv2(x)
        x = self.hrnet.bn2(x)
        x = self.hrnet.relu(x)
        x = self.hrnet.layer1(x)

        # Stage 2
        x_list = []
        for i in range(self.hrnet.stage2_cfg['NUM_BRANCHES']):
            if self.hrnet.transition1[i] is not None:
                x_list.append(self.hrnet.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.hrnet.stage2(x_list)

        # Stage 3
        x_list = []
        for i in range(self.hrnet.stage3_cfg['NUM_BRANCHES']):
            if self.hrnet.transition2[i] is not None:
                x_list.append(self.hrnet.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.hrnet.stage3(x_list)

        # Stage 4
        x_list = []
        for i in range(self.hrnet.stage4_cfg['NUM_BRANCHES']):
            if self.hrnet.transition3[i] is not None:
                x_list.append(self.hrnet.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.hrnet.stage4(x_list)

        # y_list = [branch0(32ch, s4), branch1(64ch, s8), branch2(128ch, s16), branch3(256ch, s32)]
        b0 = y_list[0]
        b1_up = F.interpolate(
            y_list[1], scale_factor=2, mode='bilinear', align_corners=False,
        )
        b0_fused = torch.cat([b0, b1_up], dim=1)  # 96ch, stride-4
        c2 = self.proj0_fused(b0_fused)   # 96 → 128 → 256, stride-4
        c3 = self.proj1(y_list[1])   # 64 → 256 → 512, stride-8
        c4 = self.proj2(y_list[2])   # 128 → 256 → 512, stride-16

        # FSNet expects: features[0]=none, [1]=C2(256ch), [2]=C3(512ch), [3]=C4(512ch)
        return [c2, c2, c3, c4]


def build_hrnet_backbone(pretrained_path: str = "") -> nn.Module:
    """Factory function matching FSNet's build_backbone interface."""
    return HRNetBackbone(pretrained_path=pretrained_path)
