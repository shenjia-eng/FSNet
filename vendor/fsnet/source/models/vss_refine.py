"""VSSBlock spatial refine (optional, requires S3Mamba VMamba code)."""
import torch.nn as nn

from .s3mamba_bridge import build_vssblock_kwargs, import_vssblock, load_vssm_yaml


class VSSRefine(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        depth=1,
        drop_path=0.1,
        vmamba_cfg=None,
        s3mamba_root=None,
        vssblock_kwargs=None,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError('vssblock_depth must be >= 1')

        VSSBlock, Permute, norm_layers, act_layers = import_vssblock(s3mamba_root)
        if vssblock_kwargs is None:
            cfg = load_vssm_yaml(vmamba_cfg)
            vssblock_kwargs = build_vssblock_kwargs(cfg)

        norm_name = str(vssblock_kwargs.get('norm_layer', 'ln2d')).lower()
        norm_layer = norm_layers.get(norm_name, norm_layers['ln2d'])
        ssm_act_layer = act_layers.get(str(vssblock_kwargs['ssm_act_layer']).lower(), nn.SiLU)
        mlp_act_layer = act_layers.get(str(vssblock_kwargs['mlp_act_layer']).lower(), nn.GELU)
        channel_first = norm_name in ('bn', 'ln2d')

        blocks = []
        for _ in range(depth):
            blocks.append(
                nn.Sequential(
                    Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
                    VSSBlock(
                        hidden_dim=hidden_dim,
                        drop_path=drop_path,
                        norm_layer=norm_layer,
                        channel_first=channel_first,
                        ssm_act_layer=ssm_act_layer,
                        mlp_act_layer=mlp_act_layer,
                        **{k: v for k, v in vssblock_kwargs.items()
                           if k not in ('norm_layer', 'ssm_act_layer', 'mlp_act_layer')},
                    ),
                    Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
