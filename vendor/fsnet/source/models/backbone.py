# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import os

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

import models.vgg_ as models


def _build_vgg_blocks(name: str):
    if name == 'vgg16_bn':
        backbone = models.vgg16_bn(pretrained=pretrained)
        features = list(backbone.features.children())
        return [
            nn.Sequential(*features[:13]),
            nn.Sequential(*features[13:23]),
            nn.Sequential(*features[23:33]),
            nn.Sequential(*features[33:43]),
        ]
    if name == 'vgg16':
        backbone = models.vgg16(pretrained=pretrained)
        features = list(backbone.features.children())
        return [
            nn.Sequential(*features[:9]),
            nn.Sequential(*features[9:16]),
            nn.Sequential(*features[16:23]),
            nn.Sequential(*features[23:30]),
        ]
    raise ValueError(f'Unsupported VGG backbone for fusion: {name}')


class CompetitiveFusionBackbone(nn.Module):
    """
    Returns (features, aux_dict) where aux_dict includes competition regularizer.
    """
    def __init__(
        self,
        vgg_name: str,
        d2_cfg_yaml: str,
        d2_freeze_at: int = 0,
        p2p_dominance_init: float = 0.85,
        competition_margin: float = 0.05,
    ):
        super().__init__()
        from models.backbone_detectron2 import build_detectron2_fpn_backbone

        self.vgg_blocks = nn.ModuleList(_build_vgg_blocks(vgg_name))
        self.d2_backbone = build_detectron2_fpn_backbone(d2_cfg_yaml, freeze_at=d2_freeze_at)
        with torch.no_grad():
            probe = torch.zeros(1, 3, 256, 256)
            vgg_probe_feats = self._extract_vgg_features(probe)
        vgg_channels = [int(feat.shape[1]) for feat in vgg_probe_feats]
        target_channels = [256, 256, 512, 512]
        self.vgg_proj = nn.ModuleList(
            [nn.Conv2d(cin, cout, kernel_size=1, bias=True) for cin, cout in zip(vgg_channels, target_channels)]
        )
        for layer in self.vgg_proj:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        dominance_init = float(max(0.51, min(0.99, p2p_dominance_init)))
        self.fusion_logit = nn.Parameter(torch.logit(torch.tensor(dominance_init)))
        self.competition_margin = float(max(0.0, competition_margin))

    def _extract_vgg_features(self, x):
        out = []
        cur = x
        for block in self.vgg_blocks:
            cur = block(cur)
            out.append(cur)
        return out

    def forward(self, tensor_list):
        vgg_feats_raw = self._extract_vgg_features(tensor_list)
        vgg_feats = [proj(feat) for proj, feat in zip(self.vgg_proj, vgg_feats_raw)]
        d2_feats = self.d2_backbone(tensor_list)
        aligned_vgg_feats = []
        for vf, df in zip(vgg_feats, d2_feats):
            if vf.shape[-2:] != df.shape[-2:]:
                vf = F.interpolate(vf, size=df.shape[-2:], mode='bilinear', align_corners=False)
            aligned_vgg_feats.append(vf)
        alpha = torch.sigmoid(self.fusion_logit)
        fused_feats = [alpha * vf + (1.0 - alpha) * df for vf, df in zip(aligned_vgg_feats, d2_feats)]

        vgg_score = torch.stack([feat.abs().mean() for feat in aligned_vgg_feats]).mean()
        d2_score = torch.stack([feat.abs().mean() for feat in d2_feats]).mean()
        competition_penalty = F.relu((1.0 - alpha) * d2_score - alpha * vgg_score + self.competition_margin)
        aux = {
            'fusion_alpha': alpha,
            'competition_penalty': competition_penalty,
        }
        return fused_feats, aux


class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
            elif name == 'vgg16':
                self.body = nn.Sequential(*features[:30])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, tensor_list):
        out = []

        if self.return_interm_layers:
            xs = tensor_list
            for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                out.append(xs)

        else:
            xs = self.body(tensor_list)
            out.append(xs)
        return out


class Backbone_VGG(BackboneBase_VGG):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, return_interm_layers: bool, pretrained=True):
        if name == 'vgg16_bn':
            backbone = models.vgg16_bn(pretrained=pretrained)
        elif name == 'vgg16':
            backbone = models.vgg16(pretrained=pretrained)
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


def build_backbone(args):
    name = getattr(args, "backbone", "vgg16_bn")
    if name in ("vmamba_tiny", "vmamba"):
        from models.vmamba_backbone import VMambaBackbone

        return VMambaBackbone(
            cfg_path=getattr(args, "vmamba_cfg", None),
            pretrained=getattr(args, "vmamba_pretrained", None),
            s3mamba_root=getattr(args, "s3mamba_root", None),
        )
    if name == "fusion_competitive":
        yaml_path = getattr(
            args,
            "detectron2_cfg",
            os.environ.get(
                "DETECTRON2_CFG",
                os.path.join(r"optional/detectron2", "configs", "Base-RCNN-FPN.yaml"),
            ),
        )
        return CompetitiveFusionBackbone(
            vgg_name=getattr(args, "fusion_p2p_backbone", "vgg16_bn"),
            d2_cfg_yaml=yaml_path,
            d2_freeze_at=int(getattr(args, "d2_freeze_at", 0)),
            p2p_dominance_init=float(getattr(args, "fusion_p2p_dominance_init", 0.85)),
            competition_margin=float(getattr(args, "fusion_competition_margin", 0.05)),
        )
    if isinstance(name, str) and name.startswith("d2_"):
        from models.backbone_detectron2 import build_detectron2_fpn_backbone

        yaml_path = getattr(
            args,
            "detectron2_cfg",
            os.environ.get(
                "DETECTRON2_CFG",
                os.path.join(r"optional/detectron2", "configs", "Base-RCNN-FPN.yaml"),
            ),
        )
        freeze_at = int(getattr(args, "d2_freeze_at", 0))
        return build_detectron2_fpn_backbone(yaml_path, freeze_at=freeze_at)
    if name.startswith("hrnet"):
        from models.backbone_hrnet import build_hrnet_backbone
        pretrained = getattr(args, "hrnet_pretrained", r"weights/hrnet_w32-36af842e.pth")
        return build_hrnet_backbone(pretrained_path=pretrained)
    backbone = Backbone_VGG(name, True, pretrained=getattr(args, 'backbone_pretrained', True))
    return backbone

if __name__ == '__main__':
    Backbone_VGG('vgg16', True)