"""
Optional Detectron2 ResNet-FPN backbone for FSNet.

Requires a working detectron2 install, e.g.:
  pip install -e optional/detectron2

Input: same as the rest of this repo — ImageNet-normalized RGB (N,3,H,W) from
torchvision transforms. Internally converted to Detectron2-style BGR (0–255) with
MODEL.PIXEL_MEAN subtracted before ResNet.

Output: list of 4 tensors compatible with models/fsnet.py + Decoder(256,512,512):
  [p3, p3, proj(p4), proj(p5)]  — indices [1],[2],[3] feed the FPN decoder.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _imagenet_to_d2_bgr_centered(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet norm -> uint8-scale BGR -> subtract Detectron2 PIXEL_MEAN."""
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb255 = (x * std + mean) * 255.0
    bgr255 = rgb255[:, [2, 1, 0], :, :]
    d2_mean = x.new_tensor([103.530, 116.280, 123.675]).view(1, 3, 1, 1)
    return bgr255 - d2_mean


class BackboneDetectron2FPN(nn.Module):
    def __init__(self, cfg_yaml: str, freeze_at: int = 0):
        super().__init__()
        try:
            from detectron2.config import get_cfg
            from detectron2.layers import ShapeSpec
            from detectron2.modeling import build_backbone as d2_build_backbone
        except ImportError as e:
            raise ImportError(
                "detectron2 is not installed. From the repo root run:\n"
                "  pip install -e optional/detectron2\n"
                "(Use your actual detectron2 path; match PyTorch/CUDA versions.)"
            ) from e

        if not os.path.isfile(cfg_yaml):
            raise FileNotFoundError(
                f"Detectron2 config not found: {cfg_yaml}\n"
                "Set --detectron2_cfg or env DETECTRON2_CFG to Base-RCNN-FPN.yaml under your detectron2 repo."
            )

        cfg = get_cfg()
        cfg.merge_from_file(cfg_yaml)
        cfg.MODEL.BACKBONE.FREEZE_AT = int(freeze_at)

        input_shape = ShapeSpec(channels=3, height=None, width=None)
        self.body = d2_build_backbone(cfg, input_shape)
        # FPN outputs 256 ch on p3,p4,p5; FSNet Decoder expects 256,512,512 for C3,C4,C5
        self.proj4 = nn.Conv2d(256, 512, kernel_size=1, bias=True)
        self.proj5 = nn.Conv2d(256, 512, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.proj4.weight)
        nn.init.xavier_uniform_(self.proj5.weight)
        nn.init.zeros_(self.proj4.bias)
        nn.init.zeros_(self.proj5.bias)

    def forward(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(f"Expected NCHW tensor, got shape {tuple(x.shape)}")
        x = _imagenet_to_d2_bgr_centered(x)
        _, _, h, w = x.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        feats = self.body(x)
        p3 = feats["p3"]
        p4 = self.proj4(feats["p4"])
        p5 = self.proj5(feats["p5"])
        return [p3, p3, p4, p5]


def build_detectron2_fpn_backbone(cfg_yaml: str, freeze_at: int = 0) -> nn.Module:
    return BackboneDetectron2FPN(cfg_yaml, freeze_at=freeze_at)
