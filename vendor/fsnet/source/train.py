import argparse
import datetime
import random
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np

from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')


def _safe_print(*args, **kwargs):
    """Windows: avoid OSError when stdout is redirected (Tee-Object / pipe)."""
    kwargs.pop("flush", None)
    try:
        print(*args, **kwargs)
    except OSError:
        pass


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training FSNet', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument(
        '--lr_proj',
        default=1e-4,
        type=float,
        help='Learning rate for backbone->decoder projection layers (e.g. HRNet proj0/1/2).',
    )
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument(
        '--backbone',
        default='vgg16_bn',
        type=str,
        help="Backbone: vgg16_bn (default), vgg16, vmamba_tiny, hrnet_w32, d2_r50_fpn, or fusion_competitive.",
    )
    parser.add_argument(
        '--hrnet_pretrained',
        default=r'weights/hrnet_w32-36af842e.pth',
        type=str,
        help='ImageNet pretrained HRNet checkpoint (hrnet_w32 / hrnet_* backbones).',
    )
    parser.add_argument(
        '--s3mamba_root',
        default=os.environ.get('S3MAMBA_ROOT', r'optional/S3Mamba'),
        type=str,
        help='Path to S3Mamba repo root (for vmamba_tiny / --use_vssblock).',
    )
    parser.add_argument(
        '--vmamba_cfg',
        default=os.path.join(
            os.environ.get('S3MAMBA_ROOT', r'optional/S3Mamba'),
            'configs',
            'vssm1',
            'vssm_tiny_224_0229flex.yaml',
        ),
        type=str,
        help='VMamba yaml config used by vmamba_tiny backbone and VSSBlock.',
    )
    parser.add_argument(
        '--vmamba_pretrained',
        default=None,
        type=str,
        help='Optional VMamba pretrained checkpoint path (vmamba_tiny only).',
    )
    parser.add_argument(
        '--detectron2_cfg',
        default=os.environ.get(
            'DETECTRON2_CFG',
            os.path.join(r'optional/detectron2', 'configs', 'Base-RCNN-FPN.yaml'),
        ),
        type=str,
        help='YAML merged for Detectron2 backbone (default: optional/detectron2\\configs\\Base-RCNN-FPN.yaml)',
    )
    parser.add_argument(
        '--d2_freeze_at',
        default=0,
        type=int,
        help='Detectron2 BACKBONE.FREEZE_AT (0 = train all ResNet stages)',
    )
    parser.add_argument(
        '--fusion_p2p_backbone',
        default='vgg16_bn',
        type=str,
        help='When --backbone fusion_competitive: FSNet main branch backbone (vgg16_bn/vgg16).',
    )
    parser.add_argument(
        '--fusion_p2p_dominance_init',
        default=0.85,
        type=float,
    )
    parser.add_argument(
        '--fusion_competition_margin',
        default=0.05,
        type=float,
        help='Margin used by competition regularizer in fusion_competitive.',
    )
    parser.add_argument(
        '--no_spatial_diff',
        dest='use_spatial_diff',
        action='store_false',
        help='Disable 2D spatial high-pass stem on RGB input.',
    )
    parser.add_argument(
        '--no_high_res_fpn',
        dest='use_high_res_fpn',
        action='store_false',
        help='Use P4-only head instead of fused P3+P4 (stride-8).',
    )
    parser.add_argument(
        '--use_ffparser',
        action='store_true',
        help='Enable FFT-based FFParser (see --ffparser_mode).',
    )
    parser.add_argument(
        '--no_ffparser',
        dest='use_ffparser',
        action='store_false',
        help='Disable FFParser (default).',
    )
    parser.add_argument(
        '--ffparser_mode',
        default='det',
        choices=['det', 'fpn', 'both'],
        help="FFParser placement: det=det_feat; fpn=P3/P4/P5; both=both.",
    )
    parser.add_argument(
        '--use_amp_phase',
        action='store_true',
        help='Replace standard FFParser with Amp-Phase decomposition + HF phase guidance.',
    )
    parser.add_argument(
        '--use_freq_spatial_gate',
        action='store_true',
        help='Enable FreqSpatialGate: P3 modulates fused det path (serial mode).',
    )
    parser.add_argument(
        '--gate_position',
        default='after_det_ff',
        choices=['before_det_ff', 'after_det_ff'],
        help='FreqSpatialGate placement: before_det_ff or after_det_ff.',
    )
    parser.add_argument(
        '--gate_mode',
        default='mul',
        choices=['mul', 'add'],
        help='Gate fusion: mul or add (residual).',
    )
    parser.add_argument(
        '--use_neck_se',
        action='store_true',
        help='Serial mode: SE channel attention on fused det_feat (before det-FF).',
    )
    parser.add_argument(
        '--use_p5_context',
        action='store_true',
        help='Serial mode: P5 GAP+MLP global context broadcast to fused det_feat.',
    )
    parser.add_argument(
        '--use_freq_attn',
        action='store_true',
        help='FFParserBlock: rFFT-amplitude SE attention after irfft, before residual add.',
    )
    parser.add_argument(
        '--attn_reduction',
        default=16,
        type=int,
        help='Channel reduction for FreqAmpChannelAttention (--use_freq_attn).',
    )
    parser.add_argument(
        '--use_det_freq_attn',
        action='store_true',
        help='SEChannelAttention on det_feat after det-FF, before gate (serial).',
    )
    parser.add_argument(
        '--det_freq_attn_reduction',
        default=16,
        type=int,
        help='Channel reduction for det SE (--use_det_freq_attn).',
    )
    parser.add_argument(
        '--no_det_freq_attn_identity_init',
        dest='det_freq_attn_identity_init',
        action='store_false',
        help='Random SE initialization (default: identity initialization).',
    )
    parser.add_argument(
        '--det_freq_attn_position',
        default='after_det_ff',
        choices=['after_det_ff', 'after_gate'],
        help='SE placement: after det-FF before gate, or after gate.',
    )
    parser.add_argument(
        '--use_det_dual_freq_attn',
        action='store_true',
        help='Parallel GAP-SE + rFFT-SE with learnable beta on det_feat.',
    )
    parser.add_argument(
        '--det_dual_freq_beta',
        default=0.1,
        type=float,
        help='Initial beta for dual frequency attention branches.',
    )
    parser.add_argument(
        '--ffparser_num_bands',
        default=1,
        type=int,
        choices=[1, 2, 3],
        help='FFParser radial band count ablation: 1=full-band (default), 2=low+high, 3=low+mid+high.',
    )
    parser.add_argument(
        '--ffparser_strategy',
        default='complex',
        choices=['none', 'complex', 'real_static', 'dynamic', 'afno_mlp', 'no_param_fft',
                 'multi_band', 'amplitude_only', 'channel_shared'],
        help='FFParser learning strategy.  Literature-inspired mechanisms: '
             'none (identity branch), '
             'complex (GFNet-style complex filtering), '
             'real_static (GFNet-style real-valued filtering), '
             'dynamic (DFFormer-style input-adaptive band modulation), '
             'afno_mlp (AFNO-inspired shared channel MLP in spectral domain), '
             'no_param_fft (FNet-style IFFT(Re(FFT(X))), zero params). '
             'multi_band, amplitude_only, channel_shared.',
    )
    parser.add_argument(
        '--dynamic_num_bands',
        default=2,
        type=int,
        choices=[1, 2, 3],
        help='Number of radial bands for --ffparser_strategy dynamic. Default 2 (low/high).',
    )
    parser.add_argument(
        '--dynamic_mlp_hidden',
        default=64,
        type=int,
        help='Hidden width for the dynamic filter generator MLP.',
    )
    parser.add_argument(
        '--afno_reduction',
        default=4,
        type=int,
        help='Channel reduction ratio for --ffparser_strategy afno_mlp.',
    )
    parser.add_argument(
        '--spatial_stem_type',
        default='laplacian',
        choices=['laplacian', 'plain', 'sobel', 'scharr', 'dog', 'log', 'fixed_laplacian'],
        help='SDS stem operator: laplacian, plain, sobel, scharr, dog, log, fixed_laplacian (frozen discrete Laplacian).',
    )
    parser.add_argument(
        '--stem_kernel_size',
        default=7,
        type=int,
        choices=[3, 7],
        help='SDS stem conv kernel size (3 or 7). Default 7.',
    )
    parser.add_argument(
        '--stem_freeze_diff',
        action='store_true',
        default=False,
        help='Freeze SDS stem diff.* weights after init (BN + 1x1 proj remain learnable).',
    )
    parser.add_argument(
        '--p5_pool_type',
        default='avg',
        choices=['avg', 'max'],
        help='P5GlobalContext pooling: adaptive avg or max.',
    )
    parser.add_argument(
        '--det_fuse_levels',
        default='p3p4',
        choices=['p3p4', 'p3p4p5'],
        help='Detection FPN fuse: P3+P4 (default) or P3+P4+P5 concat.',
    )
    parser.set_defaults(det_freq_attn_identity_init=True)
    parser.add_argument(
        '--amp_phase_cutoff',
        default=0.4,
        type=float,
        help='Frequency cutoff for HF/LF separation (0~1). Lower = more frequencies treated as HF.',
    )
    parser.add_argument(
        '--lr_ffparser',
        default=2e-4,
        type=float,
        help='Learning rate for FFParser params (default 2x --lr).',
    )
    parser.add_argument(
        '--freeze_backbone_epochs',
        default=0,
        type=int,
        help='Freeze backbone for the first N epochs, then unfreeze (set >= --epochs to keep frozen).',
    )
    parser.add_argument(
        '--freeze_modules',
        default='',
        type=str,
        help='Comma-separated name prefixes to freeze (e.g. spatial_stem,ffparser,serial_p5_context).',
    )
    parser.set_defaults(use_spatial_diff=True, use_high_res_fpn=True, use_ffparser=False)
    parser.add_argument(
        '--use_wavelet',
        action='store_true',
        help='Enable S3Mamba-style wavelet enhancement on det_feat.',
    )
    parser.add_argument(
        '--use_channel_mamba',
        action='store_true',
        help='Enable channel-group Mamba refine before detection head (requires mamba_ssm).',
    )
    parser.add_argument(
        '--use_fpn_smooth',
        action='store_true',
        help='Enable ResBlock smoothing on FPN P3/P4/P5 outputs.',
    )
    parser.add_argument(
        '--use_vssblock',
        action='store_true',
        help='Enable VSSBlock spatial refine before detection head (requires S3Mamba VMamba code).',
    )
    parser.add_argument(
        '--vssblock_depth',
        default=1,
        type=int,
        help='Number of VSSBlocks when --use_vssblock is set.',
    )
    parser.add_argument(
        '--fusion_mode',
        default='serial',
        choices=['serial', 'parallel', 'v2', 'crossscale', 'native_spatial'],
        help='Enhancement fusion: serial, parallel, v2 (NeckV2), crossscale (CrossScaleFF + serial det), native_spatial (per-scale FF + spatial top-down).',
    )
    parser.add_argument(
        '--fusion_vote',
        default='sigmoid',
        choices=['sigmoid', 'softmax', 'concat'],
        help='Parallel branch fusion: sigmoid (independent residual gates), softmax (weights sum to 1), or concat (concatenation + 1x1 fusion).',
    )

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)
    parser.add_argument(
        '--count_loss_coef',
        default=0.0,
        type=float,
        help='aux global count regression loss coefficient (0 disables density auxiliary branch)',
    )
    parser.add_argument(
        '--geo_loss_coef',
        default=0.0,
        type=float,
        help='geometry prior loss coefficient (0 disables geometry constraint)',
    )
    parser.add_argument(
        '--geo_k',
        default=3,
        type=int,
        help='k-NN size used by geometry prior loss',
    )
    parser.add_argument(
        '--density_adaptive_loss',
        action='store_true',
        help='Enable density-aware weighting on point regression loss.',
    )
    parser.add_argument(
        '--density_ref_count',
        default=80.0,
        type=float,
        help='Reference GT count for density-aware loss weighting.',
    )
    parser.add_argument(
        '--density_gamma',
        default=0.5,
        type=float,
        help='Exponent for density-aware weighting: weight=(gt_count/ref)^gamma.',
    )
    parser.add_argument(
        '--density_weight_min',
        default=0.6,
        type=float,
        help='Lower clip for density-aware loss weight.',
    )
    parser.add_argument(
        '--density_weight_max',
        default=1.8,
        type=float,
        help='Upper clip for density-aware loss weight.',
    )
    parser.add_argument(
        '--density_adaptive_mode',
        default='all',
        choices=['all', 'high_only'],
        help='all: scale all samples; high_only: only up-weight high-density samples.',
    )
    parser.add_argument(
        '--density_high_thresh',
        default=150.0,
        type=float,
        help='GT count threshold used in high_only mode.',
    )
    parser.add_argument(
        '--uncertainty_weighting',
        action='store_true',
        help='Enable learnable uncertainty weighting for multi-loss balancing.',
    )
    parser.add_argument(
        '--competition_loss_coef',
        default=0.1,
        type=float,
        help='Coefficient for competitive fusion regularization loss (set 0 to disable).',
    )

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    # dataset parameters
    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='data/MT-Set',
                        help='Single SHHA root (train.list / test.list). Ignored as the only source if --data_roots is set.')
    parser.add_argument(
        '--data_roots',
        default='',
        type=str,
        help='Merge multiple SHHA folders: paths separated by | (each folder needs train.list and test.list). '
             'Example: data/site_a|data/site_b. Overrides using only --data_root.',
    )
    parser.add_argument(
        '--train_no_patch',
        action='store_true',
        help='SHHA training: disable random patch multi-crop; use full (e.g. 800×800) images like validation. '
             'Usually lower --batch_size to avoid OOM.',
    )
    parser.add_argument(
        '--patch_side',
        default=128,
        type=int,
        help='SHHA training patch size (S×S random crop per image when not --train_no_patch). Default 128.',
    )
    parser.add_argument(
        '--shha_fixed_side',
        default=0,
        type=int,
        help='SHHA: if >0, resize images to S×S with bilinear interpolation and scale point coordinates; skip images already S×S. '
             'Use 0 to disable.',
    )

    parser.add_argument('--output_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./ckpt',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--latest_checkpoint_name', default='latest.pth', type=str,
                        help='filename for latest checkpoint under checkpoints_dir')
    parser.add_argument('--best_checkpoint_name', default='best_mae.pth', type=str,
                        help='filename for best-MAE checkpoint under checkpoints_dir')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint (model + optimizer/epoch if present)')
    parser.add_argument('--init_weights', default='', type=str,
                        help='load model weights only; optimizer/LR schedule reset (continual learning / fine-tune). '
                             'Do not use together with --resume.')
    parser.add_argument(
        '--init_weights_non_backbone',
        action='store_true',
        help='With --init_weights: load only keys not starting with "backbone." (shape must match). '
             'Use when switching backbone e.g. vgg checkpoint -> d2_r50_fpn: FSNet heads/FPN from old ckpt, '
             'Detectron2 backbone from its own init.',
    )
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument(
        '--val_sliding_for_best',
        action='store_true',
        help='Use sliding-window MAE (not whole-image) to save best checkpoint. '
             'Whole-image val still runs each eval_freq for logging.',
    )
    parser.add_argument(
        '--sliding_eval_freq',
        default=25,
        type=int,
        help='When --val_sliding_for_best: run sliding val every N epochs to update best ckpt '
             '(whole-image val still every --eval_freq).',
    )
    parser.add_argument('--sliding_window', default=128, type=int)
    parser.add_argument('--sliding_stride', default=64, type=int)
    parser.add_argument('--sliding_merge_radius', default=8.0, type=float)
    parser.add_argument('--sliding_batch_tiles', default=16, type=int)
    parser.add_argument('--val_score_threshold', default=0.5, type=float)
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')
    parser.add_argument(
        '--log_batch_interval',
        default=0,
        type=int,
        help='若>0：每 N 个训练 iteration 向终端打印一次 loss。0 关闭。',
    )
    parser.add_argument(
        '--no_plot_report',
        action='store_true',
        help='训练结束后不自动生成 loss/指标曲线与 training_summary（默认会生成）。',
    )
    parser.add_argument(
        '--no_post_eval_best',
        action='store_true',
        help='生成报告时不额外用 best 权重在验证集上再评估一次。',
    )

    return parser


def _resolve_data_roots(args):
    raw = (getattr(args, "data_roots", None) or "").strip()
    if raw:
        roots = [x.strip() for x in raw.split("|") if x.strip()]
        if not roots:
            raise ValueError("--data_roots is empty after parsing")
        return roots
    return [args.data_root]


def _set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = trainable


def _param_matches_freeze_prefix(name: str, prefix: str) -> bool:
    prefix = prefix.strip()
    if not prefix:
        return False
    if prefix == 'ffparser':
        return 'ffparser' in name
    return name.startswith(prefix + '.') or name == prefix


def _set_modules_trainable_by_prefixes(model, prefixes: list[str], trainable: bool) -> int:
    if not prefixes:
        return 0
    n = 0
    for name, param in model.named_parameters():
        if any(_param_matches_freeze_prefix(name, p) for p in prefixes):
            param.requires_grad = trainable
            n += 1
    return n


def _apply_freeze_modules(model, freeze_modules: str) -> None:
    prefixes = [p.strip() for p in (freeze_modules or '').split(',') if p.strip()]
    if not prefixes:
        return
    n = _set_modules_trainable_by_prefixes(model, prefixes, False)
    print('freeze_modules=%r: froze %d parameter tensors' % (prefixes, n))


def _interpolate_ffparser_complex_weight(ckpt_weight, target_shape):
    """Resize stored FFParser freq weights to match the built model."""
    target_h, target_w = target_shape[1], target_shape[2]
    old = torch.view_as_complex(ckpt_weight.detach().float())
    real = F.interpolate(
        old.real.unsqueeze(1),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False,
    ).squeeze(1)
    imag = F.interpolate(
        old.imag.unsqueeze(1),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False,
    ).squeeze(1)
    return torch.view_as_real(torch.complex(real, imag)).to(dtype=ckpt_weight.dtype)


def _remap_ffparser_checkpoint_key(k, v, model_sd):
    """Map single-band FFParser weights onto multi-band modules."""
    if k.endswith('.ff.complex_weight') and k in model_sd:
        if model_sd[k].shape == v.shape:
            return k, v
        band0 = k.replace('.ff.complex_weight', '.ff.band_weights.0')
        if band0 in model_sd and model_sd[band0].shape == v.shape:
            return band0, v
    if '.ff.complex_weight' in k:
        prefix = k[: k.index('.ff.complex_weight')]
        band0 = prefix + '.ff.band_weights.0'
        if band0 in model_sd and model_sd[band0].shape == v.shape:
            return band0, v
        if band0 in model_sd:
            return band0, _interpolate_ffparser_complex_weight(v, model_sd[band0].shape)
    return None


def _load_checkpoint_state(model, ckpt_state):
    model_sd = model.state_dict()
    to_load = {}
    skipped = []
    for k, v in ckpt_state.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            to_load[k] = v
            continue
        if k.endswith('.ff.complex_weight') and 'ffparser' in k:
            if k in model_sd and model_sd[k].shape != v.shape:
                to_load[k] = _interpolate_ffparser_complex_weight(v, model_sd[k].shape)
                print(
                    'Interpolated FFParser weight %s: %s -> %s'
                    % (k, tuple(v.shape), tuple(model_sd[k].shape))
                )
                continue
            remapped = _remap_ffparser_checkpoint_key(k, v, model_sd)
            if remapped is not None:
                rk, rv = remapped
                to_load[rk] = rv
                print('Remapped FFParser weight %s -> %s' % (k, rk))
                continue
        if k not in model_sd:
            continue
        if k.endswith('.ff.complex_weight') and 'ffparser' in k:
            to_load[k] = _interpolate_ffparser_complex_weight(v, model_sd[k].shape)
            print(
                'Interpolated FFParser weight %s: %s -> %s'
                % (k, tuple(v.shape), tuple(model_sd[k].shape))
            )
        else:
            skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
    incompat = model.load_state_dict(to_load, strict=False)
    if skipped:
        print('Skipped shape-mismatch keys:', skipped)
    return incompat


def _prime_ffparser_from_batch(model, data_loader, device):
    blocks = [
        getattr(model, name, None)
        for name in ('ffparser', 'ffparser_p3', 'ffparser_p4', 'ffparser_p5')
    ]
    blocks = [b for b in blocks if b is not None]
    if not blocks:
        return
    if all(
        hasattr(b.ff, 'complex_weight') or len(getattr(b.ff, 'band_weights', [])) > 0
        for b in blocks
    ):
        return
    batch = next(iter(data_loader))
    samples = batch[0]
    if hasattr(samples, 'to'):
        samples = samples.to(device)
    with torch.no_grad():
        model(samples)
    for name in ('ffparser', 'ffparser_p3', 'ffparser_p4', 'ffparser_p5'):
        block = getattr(model, name, None)
        if block is not None and hasattr(block.ff, 'complex_weight'):
            print(
                'FFParser %s: initialized, weight shape=%s'
                % (name, tuple(block.ff.complex_weight.shape))
            )


def _build_optimizer_param_groups(model, criterion, args):
    ffparser_params = []
    head_params = []
    backbone_params = []
    proj_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'ffparser' in name:
            ffparser_params.append(param)
        elif name.startswith('backbone.proj'):
            proj_params.append(param)
        elif 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_dicts = [
        {"params": head_params, "lr": args.lr},
    ]
    if backbone_params:
        param_dicts.append({"params": backbone_params, "lr": args.lr_backbone})
    if proj_params:
        param_dicts.append({"params": proj_params, "lr": args.lr_proj})
    if ffparser_params:
        param_dicts.append({"params": ffparser_params, "lr": args.lr_ffparser})

    criterion_params = [p for p in criterion.parameters() if p.requires_grad]
    if criterion_params:
        param_dicts.append({"params": criterion_params, "lr": args.lr})
    return param_dicts


def main(args):
    raise RuntimeError('Entry disabled; use scripts/train.py')
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    # create / append the logging file (append when resuming so prior log is kept)
    run_log_name = os.path.join(args.output_dir, 'run_log.txt')
    _resume_path = (getattr(args, 'resume', None) or '').strip()
    log_append = bool(_resume_path and os.path.isfile(run_log_name))
    with open(run_log_name, "a" if log_append else "w", encoding="utf-8") as log_file:
        if log_append:
            log_file.write("\n\n--- resumed training %s ---\n" % time.strftime("%c"))
        else:
            log_file.write('Eval Log %s\n' % time.strftime("%c"))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    # backup the arguments
    print(args)
    with open(run_log_name, "a", encoding="utf-8") as log_file:
        log_file.write("{}\n".format(args))
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, criterion = build_model(args, training=True)
    # move to GPU
    model.to(device)
    criterion.to(device)

    model_without_ddp = model

    if args.freeze_backbone_epochs > 0:
        _set_backbone_trainable(model_without_ddp, False)
        print(
            'Backbone frozen for first %d epoch(s)'
            % args.freeze_backbone_epochs
        )

    _apply_freeze_modules(model_without_ddp, getattr(args, 'freeze_modules', ''))

    # create the dataset (before optimizer when FFParser may lazy-init)
    loading_data = build_dataset(args=args)
    data_roots = _resolve_data_roots(args)
    train_set, val_set = loading_data(data_roots)
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    if len(train_set) == 0:
        raise RuntimeError(f"Train set is empty. Check train.list under: {data_roots}")
    if len(data_loader_train) == 0:
        raise RuntimeError(
            f"Train DataLoader has 0 batches (train_set={len(train_set)}, batch_size={args.batch_size}, drop_last=True). "
            "Lower batch_size or add more training images."
        )

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    if args.init_weights and args.resume:
        raise RuntimeError('Use either --init_weights or --resume, not both.')
    if args.init_weights:
        checkpoint = torch.load(args.init_weights, map_location='cpu')
        ckpt_model = checkpoint['model']
        if args.init_weights_non_backbone:
            model_sd = model_without_ddp.state_dict()
            to_load = {
                k: v
                for k, v in ckpt_model.items()
                if not k.startswith('backbone.')
                and k in model_sd
                and model_sd[k].shape == v.shape
            }
            skipped_nb = [k for k in ckpt_model if not k.startswith('backbone.') and k not in to_load]
            incompat = _load_checkpoint_state(model_without_ddp, to_load)
            print(
                'Loaded init_weights_non_backbone: %d tensors; skipped non-backbone (missing/shape mismatch): %d; start_epoch=%d'
                % (len(to_load), len(skipped_nb), args.start_epoch)
            )
        else:
            incompat = _load_checkpoint_state(model_without_ddp, ckpt_model)
            print(
                'Loaded init_weights (model only), start_epoch=%d, missing=%d, unexpected=%d'
                % (args.start_epoch, len(incompat.missing_keys), len(incompat.unexpected_keys))
            )
    elif args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        incompat = _load_checkpoint_state(model_without_ddp, checkpoint['model'])
        print('Resume load_state_dict: missing=%d unexpected=%d' % (len(incompat.missing_keys), len(incompat.unexpected_keys)))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            args._resume_optimizer = checkpoint['optimizer']
            args._resume_lr_scheduler = checkpoint['lr_scheduler']
            args.start_epoch = checkpoint['epoch'] + 1
            print('Resumed from epoch %d' % checkpoint['epoch'])
        else:
            print('Checkpoint has model only; optimizer/epoch reset, start_epoch=%d' % args.start_epoch)

    if args.use_ffparser:
        _prime_ffparser_from_batch(model_without_ddp, data_loader_train, device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if args.use_ffparser:
        print(
            'FFParser enabled mode=%s (lr_ffparser=%.2e)'
            % (args.ffparser_mode, args.lr_ffparser)
        )
    param_dicts = _build_optimizer_param_groups(model_without_ddp, criterion, args)
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    if getattr(args, '_resume_optimizer', None) is not None and not args.eval:
        resume_opt = args._resume_optimizer
        if len(optimizer.param_groups) == len(resume_opt.get('param_groups', [])):
            optimizer.load_state_dict(resume_opt)
            lr_scheduler.load_state_dict(args._resume_lr_scheduler)
        else:
            print(
                'Optimizer param groups changed (e.g. backbone unfrozen); '
                'using fresh optimizer from epoch %d' % args.start_epoch
            )

    print("Start training")
    with open(run_log_name, "a", encoding="utf-8") as log_file:
        log_file.write("\nStart training\n")
    start_time = time.time()
    # save validation metrics during training (test.list, not train.list)
    mae = []
    best_sliding_mae = float("inf")
    best_sliding_epoch = -1
    # the logger writer
    writer = SummaryWriter(args.tensorboard_dir)
    
    step = 0
    # training starts here
    for epoch in range(args.start_epoch, args.epochs):
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs:
            _set_backbone_trainable(model_without_ddp, True)
            optimizer = torch.optim.Adam(
                _build_optimizer_param_groups(model_without_ddp, criterion, args),
                lr=args.lr,
            )
            print('Backbone unfrozen at epoch %d' % epoch)

        t1 = time.time()
        stat = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            log_batch_interval=int(getattr(args, "log_batch_interval", 0) or 0),
        )

        # record the training states after every epoch
        if writer is not None:
            with open(run_log_name, "a", encoding="utf-8") as log_file:
                log_file.write(
                    "loss/loss@{}: {}\nloss/loss_ce@{}: {}\n".format(
                        epoch, stat["loss"], epoch, stat["loss_ce"]
                    )
                )
                for k, v in sorted(stat.items()):
                    if k in ("loss", "loss_ce", "lr") or not k.startswith("loss_"):
                        continue
                    log_file.write("loss/{}@{}: {}\n".format(k, epoch, v))
            writer.add_scalar('loss/loss', stat['loss'], epoch)
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)

        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        with open(run_log_name, "a", encoding="utf-8") as log_file:
            log_file.write(
                "[ep %d][lr %.7f][%.2fs]\n"
                % (epoch, optimizer.param_groups[0]["lr"], t2 - t1)
            )
        # change lr according to the scheduler
        lr_scheduler.step()
        # save latest weights every epoch
        checkpoint_latest_path = os.path.join(args.checkpoints_dir, args.latest_checkpoint_name)
        torch.save({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, checkpoint_latest_path)
        # run evaluation
        if epoch % args.eval_freq == 0 and epoch != 0:
            val_threshold = float(getattr(args, "val_score_threshold", 0.5) or 0.5)
            t1 = time.time()
            val_metrics = evaluate_crowd_count_metrics(
                model, data_loader_val, device, threshold=val_threshold
            )
            t2 = time.time()

            mae.append(val_metrics['mae'])
            best_whole_mae_so_far = float(np.min(mae))
            use_sliding_best = bool(getattr(args, "val_sliding_for_best", False))
            sliding_freq = max(1, int(getattr(args, "sliding_eval_freq", 25) or 25))
            run_sliding = use_sliding_best and (epoch % sliding_freq == 0)
            slide_metrics = None
            slide_cfg = None
            t_slide_start = t2
            t_slide_end = t2

            if run_sliding:
                from tools.sliding_window_core import (
                    SlidingConfig,
                    evaluate_sliding_count_metrics,
                )

                slide_cfg = SlidingConfig(
                    window=int(getattr(args, "sliding_window", 128) or 128),
                    stride=int(getattr(args, "sliding_stride", 64) or 64),
                    merge_radius=float(getattr(args, "sliding_merge_radius", 8.0) or 8.0),
                    score_threshold=val_threshold,
                    batch_tiles=int(getattr(args, "sliding_batch_tiles", 16) or 16),
                )
                t_slide_start = time.time()
                _safe_print(
                    "sliding val (best ckpt): window=%d stride=%d merge=%.1f thr=%.2f tiles=%d"
                    % (
                        slide_cfg.window,
                        slide_cfg.stride,
                        slide_cfg.merge_radius,
                        slide_cfg.score_threshold,
                        slide_cfg.batch_tiles,
                    )
                )
                try:
                    slide_metrics = evaluate_sliding_count_metrics(
                        model,
                        data_loader_val,
                        device,
                        cfg=slide_cfg,
                        threshold=val_threshold,
                        progress_every=0,
                    )
                except Exception as slide_ex:
                    _safe_print("sliding val FAILED at epoch %d: %s" % (epoch, slide_ex))
                    with open(run_log_name, "a", encoding="utf-8") as log_file:
                        log_file.write(
                            "\n--- sliding val failed ep %d ---\n%s\n"
                            % (epoch, traceback.format_exc())
                        )
                    slide_metrics = None
                t_slide_end = time.time()

            sliding_improved = False
            if slide_metrics is not None:
                sliding_improved = slide_metrics["mae"] < best_sliding_mae
                if sliding_improved:
                    best_sliding_mae = float(slide_metrics["mae"])
                    best_sliding_epoch = epoch

            print('=======================================val (whole-image)=======================================')
            print(format_count_metrics(val_metrics))
            print(
                "eval_time: %.2fs  epoch: %d  best whole MAE so far: %.4f"
                % (t2 - t1, epoch, best_whole_mae_so_far)
            )
            if slide_metrics is not None:
                print('=======================================val (sliding %dx%d)=======================================' % (
                    slide_cfg.window, slide_cfg.window
                ))
                print(format_count_metrics(slide_metrics))
                tag = " NEW BEST" if sliding_improved else ""
                print(
                    "sliding_eval_time: %.2fs  MAE=%.4f  best_sliding=%.4f @ ep %d%s"
                    % (
                        t_slide_end - t_slide_start,
                        slide_metrics["mae"],
                        best_sliding_mae,
                        best_sliding_epoch,
                        tag,
                    )
                )
            elif use_sliding_best:
                next_ep = ((epoch // sliding_freq) + 1) * sliding_freq
                print(
                    "sliding val skipped (next at ep %d, sliding_eval_freq=%d)"
                    % (next_ep, sliding_freq)
                )
            print('=======================================val=======================================')
            with open(run_log_name, "a", encoding="utf-8") as log_file:
                log_validation_metrics(
                    writer, log_file, step, epoch, val_metrics, t2 - t1
                )
                log_file.write(
                    "whole_mae:{}, mse:{}, acc5:{}, acc10:{}, time:{}, best_whole_mae:{}\n".format(
                        val_metrics['mae'],
                        val_metrics['rmse'],
                        val_metrics['acc_at_count_tol'],
                        val_metrics['acc_at_ratio_tol'],
                        t2 - t1,
                        best_whole_mae_so_far,
                    )
                )
                if slide_metrics is not None:
                    log_file.write(
                        "sliding_mae:{}, sliding_mse:{}, sliding_time:{:.2f}, "
                        "best_sliding_mae:{}, best_sliding_epoch:{}\n".format(
                            slide_metrics["mae"],
                            slide_metrics["rmse"],
                            t_slide_end - t_slide_start,
                            best_sliding_mae,
                            best_sliding_epoch,
                        )
                    )
                    if writer is not None:
                        writer.add_scalar("val/mae_sliding", slide_metrics["mae"], step)
                        writer.add_scalar("val/rmse_sliding", slide_metrics["rmse"], step)
            print('=======================================val=======================================')
            step += 1

            save_best = False
            if use_sliding_best:
                save_best = sliding_improved
            elif abs(best_whole_mae_so_far - val_metrics['mae']) < 0.01:
                save_best = True

            if save_best:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, args.best_checkpoint_name)
                ckpt_meta = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }
                if use_sliding_best:
                    ckpt_meta['best_sliding_mae'] = best_sliding_mae
                    ckpt_meta['best_sliding_epoch'] = best_sliding_epoch
                torch.save(ckpt_meta, checkpoint_best_path)
                if use_sliding_best:
                    print(
                        "Saved best checkpoint (sliding MAE=%.4f @ ep %d): %s"
                        % (best_sliding_mae, best_sliding_epoch, checkpoint_best_path)
                    )
                else:
                    print("Saved best checkpoint (whole MAE=%.4f): %s" % (
                        val_metrics['mae'], checkpoint_best_path
                    ))
    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if not getattr(args, "no_plot_report", False):
        try:
            from tools.training_report import generate_training_report

            rep = generate_training_report(
                run_log_path=run_log_name,
                output_dir=args.output_dir,
                args=args,
                device=device,
                post_eval_best=not getattr(args, "no_post_eval_best", False),
            )
            plot_dir = os.path.join(args.output_dir, "plots")
            print("Training report: %s" % os.path.join(args.output_dir, "training_summary.txt"))
            print("  plots: %s" % plot_dir)
            if rep.get("post_eval_best"):
                pe = rep["post_eval_best"]
                print("  post-eval (best ckpt) on validation set:")
                print("  " + format_count_metrics(pe).replace("\n", "\n  "))
        except Exception as ex:
            print("Training report failed:", ex)
            try:
                with open(run_log_name, "a", encoding="utf-8") as log_file:
                    log_file.write("\ntraining_report failed: %s\n" % ex)
            except OSError:
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FSNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        err = traceback.format_exc()
        print(err)
        try:
            log_path = os.path.join(args.output_dir, "run_log.txt")
            os.makedirs(args.output_dir, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write("\n--- training crashed ---\n")
                log_file.write(err)
        except OSError:
            pass
        raise