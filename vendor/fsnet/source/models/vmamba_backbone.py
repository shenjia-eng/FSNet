"""Optional VMamba backbone for FSNet (--backbone vmamba_tiny)."""
import torch.nn as nn

from .s3mamba_bridge import (
    build_backbone_vssm_kwargs,
    ensure_s3mamba_path,
    load_vssm_yaml,
)


class VMambaBackbone(nn.Module):
    """
    Wraps S3Mamba Backbone_VSSM and projects stage features to the
    channel layout expected by the existing FSNet FPN (256/512/512).
    """

    FPN_CHANNELS = (256, 512, 512)

    def __init__(self, cfg_path=None, pretrained=None, s3mamba_root=None):
        super().__init__()
        ensure_s3mamba_path(s3mamba_root)
        from models.changedetection.models.Mamba_backbone import Backbone_VSSM

        cfg = load_vssm_yaml(cfg_path)
        kwargs = build_backbone_vssm_kwargs(cfg, pretrained=pretrained)
        self.vmamba = Backbone_VSSM(out_indices=(0, 1, 2, 3), **kwargs)

        encoder_dims = list(self.vmamba.dims)
        self.stage_proj = nn.ModuleList([
            nn.Conv2d(encoder_dims[1], self.FPN_CHANNELS[0], kernel_size=1, bias=True),
            nn.Conv2d(encoder_dims[2], self.FPN_CHANNELS[1], kernel_size=1, bias=True),
            nn.Conv2d(encoder_dims[3], self.FPN_CHANNELS[2], kernel_size=1, bias=True),
        ])
        for layer in self.stage_proj:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, tensor_list):
        feats = self.vmamba(tensor_list)
        out = [feats[0]]
        for idx, proj in enumerate(self.stage_proj):
            out.append(proj(feats[idx + 1]))
        return out
