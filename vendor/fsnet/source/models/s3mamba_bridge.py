"""Helpers to load VMamba / VSSBlock from the S3Mamba repo."""
import os
import sys
from typing import Any, Dict, Optional

import yaml

_DEFAULT_S3MAMBA_ROOT = os.environ.get(
    'S3MAMBA_ROOT',
    r'optional/S3Mamba',
)
_DEFAULT_VMAMBA_CFG = os.path.join(
    _DEFAULT_S3MAMBA_ROOT,
    'configs',
    'vssm1',
    'vssm_tiny_224_0229flex.yaml',
)

_VSSM_DEFAULTS: Dict[str, Any] = {
    'PATCH_SIZE': 4,
    'IN_CHANS': 3,
    'EMBED_DIM': 96,
    'DEPTHS': [2, 2, 9, 2],
    'SSM_D_STATE': 16,
    'SSM_RATIO': 2.0,
    'SSM_RANK_RATIO': 2.0,
    'SSM_DT_RANK': 'auto',
    'SSM_ACT_LAYER': 'silu',
    'SSM_CONV': 3,
    'SSM_CONV_BIAS': True,
    'SSM_DROP_RATE': 0.0,
    'SSM_INIT': 'v0',
    'SSM_FORWARDTYPE': 'v2',
    'MLP_RATIO': 4.0,
    'MLP_ACT_LAYER': 'gelu',
    'MLP_DROP_RATE': 0.0,
    'PATCH_NORM': True,
    'NORM_LAYER': 'ln2d',
    'DOWNSAMPLE': 'v2',
    'PATCHEMBED': 'v2',
    'GMLP': False,
}


def ensure_s3mamba_path(s3mamba_root: Optional[str] = None) -> str:
    root = os.path.abspath(s3mamba_root or _DEFAULT_S3MAMBA_ROOT)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            'S3Mamba root not found: %s. Set --s3mamba_root or S3MAMBA_ROOT.' % root
        )
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return root


def load_vssm_yaml(cfg_path: Optional[str] = None) -> Dict[str, Any]:
    path = cfg_path or _DEFAULT_VMAMBA_CFG
    if not os.path.isfile(path):
        raise FileNotFoundError('VMamba config not found: %s' % path)
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    model_cfg = raw.get('MODEL', {})
    vssm = dict(_VSSM_DEFAULTS)
    vssm.update(model_cfg.get('VSSM', {}) or {})
    return {
        'drop_path_rate': float(model_cfg.get('DROP_PATH_RATE', 0.1)),
        'num_classes': int(model_cfg.get('NUM_CLASSES', 1000)),
        'vssm': vssm,
        'use_checkpoint': bool(model_cfg.get('USE_CHECKPOINT', False)),
    }


def _dt_rank(vssm: Dict[str, Any]):
    value = vssm.get('SSM_DT_RANK', 'auto')
    if value == 'auto':
        return 'auto'
    return int(value)


def build_backbone_vssm_kwargs(cfg: Dict[str, Any], pretrained: Optional[str] = None) -> Dict[str, Any]:
    v = cfg['vssm']
    return {
        'pretrained': pretrained,
        'patch_size': int(v['PATCH_SIZE']),
        'in_chans': int(v['IN_CHANS']),
        'num_classes': int(cfg.get('num_classes', 1000)),
        'depths': list(v['DEPTHS']),
        'dims': int(v['EMBED_DIM']),
        'ssm_d_state': int(v['SSM_D_STATE']),
        'ssm_ratio': float(v['SSM_RATIO']),
        'ssm_rank_ratio': float(v.get('SSM_RANK_RATIO', 2.0)),
        'ssm_dt_rank': _dt_rank(v),
        'ssm_act_layer': str(v['SSM_ACT_LAYER']),
        'ssm_conv': int(v['SSM_CONV']),
        'ssm_conv_bias': bool(v['SSM_CONV_BIAS']),
        'ssm_drop_rate': float(v['SSM_DROP_RATE']),
        'ssm_init': str(v['SSM_INIT']),
        'forward_type': str(v['SSM_FORWARDTYPE']),
        'mlp_ratio': float(v['MLP_RATIO']),
        'mlp_act_layer': str(v['MLP_ACT_LAYER']),
        'mlp_drop_rate': float(v['MLP_DROP_RATE']),
        'drop_path_rate': float(cfg['drop_path_rate']),
        'patch_norm': bool(v['PATCH_NORM']),
        'norm_layer': str(v['NORM_LAYER']),
        'downsample_version': str(v['DOWNSAMPLE']),
        'patchembed_version': str(v['PATCHEMBED']),
        'gmlp': bool(v['GMLP']),
        'use_checkpoint': bool(cfg.get('use_checkpoint', False)),
    }


def build_vssblock_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    v = cfg['vssm']
    return {
        'ssm_d_state': int(v['SSM_D_STATE']),
        'ssm_ratio': float(v['SSM_RATIO']),
        'ssm_dt_rank': _dt_rank(v),
        'ssm_conv': int(v['SSM_CONV']),
        'ssm_conv_bias': bool(v['SSM_CONV_BIAS']),
        'ssm_drop_rate': float(v['SSM_DROP_RATE']),
        'ssm_init': str(v['SSM_INIT']),
        'forward_type': 'v01',
        'mlp_ratio': float(v['MLP_RATIO']),
        'mlp_drop_rate': float(v['MLP_DROP_RATE']),
        'gmlp': bool(v['GMLP']),
        'use_checkpoint': bool(cfg.get('use_checkpoint', False)),
        'norm_layer': str(v['NORM_LAYER']),
        'ssm_act_layer': str(v['SSM_ACT_LAYER']),
        'mlp_act_layer': str(v['MLP_ACT_LAYER']),
    }


def import_vssblock(s3mamba_root: Optional[str] = None):
    import importlib

    root = os.path.abspath(s3mamba_root or _DEFAULT_S3MAMBA_ROOT)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            'S3Mamba root not found: %s. Set --s3mamba_root or S3MAMBA_ROOT.' % root
        )

    p2p_models = sys.modules.get('models')
    p2p_root = None
    if p2p_models is not None and getattr(p2p_models, '__path__', None):
        p2p_root = os.path.dirname(os.path.abspath(p2p_models.__path__[0]))

    saved_modules = {
        key: sys.modules.pop(key)
        for key in list(sys.modules)
        if key == 'models' or key.startswith('models.')
    }
    saved_path = list(sys.path)
    filtered_path = [p for p in saved_path if os.path.abspath(p) not in {root, p2p_root}]
    sys.path = [root] + filtered_path
    try:
        vmamba_mod = importlib.import_module('models.classification.models.vmamba')
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
        except ImportError:
            selective_scan_fn = None
        import selective_scan_cuda
        vmamba_mod.selective_scan_cuda = selective_scan_cuda
        _ss2d_init = vmamba_mod.SS2D.__init__

        def _ss2d_init_patched(self, *args, **kwargs):
            if kwargs.get('forward_type') == 'v01':
                return vmamba_mod.SS2D.__initv2__(self, *args, **kwargs)
            return _ss2d_init(self, *args, **kwargs)

        vmamba_mod.SS2D.__init__ = _ss2d_init_patched
        VSSBlock = vmamba_mod.VSSBlock
        LayerNorm2d = vmamba_mod.LayerNorm2d
        Permute = vmamba_mod.Permute
    finally:
        sys.path = saved_path
        sys.modules.update(saved_modules)

    import torch.nn as nn

    norm_name = 'ln2d'
    _NORMLAYERS = {
        'ln': nn.LayerNorm,
        'ln2d': LayerNorm2d,
        'bn': nn.BatchNorm2d,
    }
    _ACTLAYERS = {
        'silu': nn.SiLU,
        'gelu': nn.GELU,
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
    }
    return VSSBlock, Permute, _NORMLAYERS, _ACTLAYERS
