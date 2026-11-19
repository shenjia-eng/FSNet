"""FSNet ablation configurations.

Apply after the model builder creates FFT parameters and before loading weights
or constructing the optimizer. Restore non-tensor settings, including strict_sds,
when loading a checkpoint."""
import ast
import hashlib
from functools import lru_cache
from pathlib import Path
from types import MethodType
import torch
from torch import nn
import torch.nn.functional as F

VARIANTS = ('fsnet', 'no_sds', 'strict_sds', 'no_filter', 'no_fsc', 'dfformer', 'no_gdr')
FFT_SLOTS = ('ffparser', 'ffparser_p3', 'ffparser_p4', 'ffparser_p5')
# FFT parameter grids for 128-pixel training crops.
HISTORICAL_GRID = {
    'ffparser': (256, 16, 9), 'ffparser_p3': (256, 16, 9),
    'ffparser_p4': (256, 8, 5), 'ffparser_p5': (256, 4, 3),
}
VENDOR = Path(__file__).resolve().parent / 'vendor/dfformer_upstream.py'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def upstream_dynamic_class():
    source = VENDOR.read_text()
    tree = ast.parse(source)
    wanted = {'StarReLU', 'Mlp', 'resize_complex_weight', 'DynamicFilter'}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise RuntimeError('Official DFFormer definitions are incomplete.')
    ns = {'torch': torch, 'nn': nn, 'F': F,
          'to_2tuple': lambda x: x if isinstance(x, (tuple, list)) else (x, x)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(VENDOR), 'exec'), ns)
    return ns['DynamicFilter']


class DynamicAdapter(nn.Module):
    """NHWC DynamicFilter in a residual block."""
    def __init__(self, channels, height, width):
        super().__init__()
        self.dynamic = upstream_dynamic_class()(
            channels, size=(height, width), num_filters=4, expansion_ratio=2,
            reweight_expansion_ratio=.25, weight_resize=True)

    def forward(self, x):
        return self.dynamic(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def strict_sds_forward(self, input):
    weight = self.weight
    h, w = weight.shape[-2:]
    transformed = -weight.clone()
    transformed[..., h // 2, w // 2] = weight.sum(dim=(-2, -1)) - weight[..., h // 2, w // 2]
    return F.conv2d(input, transformed, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)


def inheritance_policy(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown staged variant: ' + str(variant))
    removed = []
    replaced = []
    if variant == 'no_sds':
        removed = ['spatial_stem.']
    elif variant in ('no_filter', 'no_fsc'):
        removed = [name + '.' for name in FFT_SLOTS]
        if variant == 'no_fsc':
            removed += ['serial_p5_context.']
    elif variant == 'no_gdr':
        removed = ['det_freq_attn.', 'freq_spatial_gate.']
    elif variant == 'dfformer':
        replaced = [name + '.ff.' for name in FFT_SLOTS]
    return {
        'removed_prefixes': removed,
        'replaced_prefixes': replaced,
        'new_dynamic_prefixes': [name + '.ff.dynamic.' for name in FFT_SLOTS] if variant == 'dfformer' else [],
        'strict_sds_state_dict_keys_unchanged': variant == 'strict_sds',
        'unknown_missing_or_unexpected_keys_must_raise': True,
        'subsequent_parent_must_match_variant_and_seed': True,
        'static_to_dynamic_weights_have_no_tensor_mapping': variant == 'dfformer',
    }


def apply_variant(model, variant, seed):
    """Mutate an already materialized stage model and return JSON metadata.

    CPU RNG state is preserved. Reapplying the same (variant, seed) is idempotent;
    switching an already adapted model to a different identity is rejected.
    Stage transitions still require rebuilding then loading that variant's own
    parent checkpoint, with an explicit stage-new-module whitelist.
    """
    seed = int(seed)
    if variant not in VARIANTS:
        raise ValueError('Unknown staged variant: ' + str(variant))
    identity = (variant, seed)
    existing = getattr(model, '_staged_variant_identity', None)
    if existing is not None:
        if existing != identity:
            raise ValueError('Cannot change an adapted model variant/seed in place.')
        return dict(model._staged_variant_metadata)

    before = model.state_dict()
    before_keys = set(before)
    before_ids = {name: id(p) for name, p in model.named_parameters()}
    before_numel = sum(p.numel() for p in model.parameters())
    gamma_before = {
        name: (id(getattr(model, name).gamma), getattr(model, name).gamma.detach().clone())
        for name in FFT_SLOTS if getattr(model, name, None) is not None
    }
    dynamic_slots = []
    if variant == 'no_sds':
        model.spatial_stem = None
    elif variant == 'strict_sds':
        stem = getattr(model, 'spatial_stem', None)
        if stem is None or not hasattr(stem, 'diff'):
            raise ValueError('strict_sds requires an existing learnable spatial stem.')
        diff = stem.diff
        if type(diff).__name__ != 'SDifferenceConv2d':
            raise TypeError('strict_sds expected SDifferenceConv2d.')
        diff.forward = MethodType(strict_sds_forward, diff)
        diff.zero_sum = True
        diff._staged_strict_sds = True
    elif variant in ('no_filter', 'no_fsc'):
        for name in FFT_SLOTS:
            setattr(model, name, None)
        if variant == 'no_fsc':
            model.serial_p5_context = None
    elif variant == 'no_gdr':
        model.det_freq_attn = None
        model.freq_spatial_gate = None
    elif variant == 'dfformer':
        blocks = {name: getattr(model, name, None) for name in FFT_SLOTS}
        grids = dict(HISTORICAL_GRID)
        for name, block in blocks.items():
            if block is None:
                continue
            if getattr(block, 'strategy', None) != 'complex':
                raise ValueError('DFFormer replacement requires complex FFT residual blocks: ' + name)
            if isinstance(block.ff, DynamicAdapter):
                continue
            weight = getattr(block.ff, 'complex_weight', None)
            if not isinstance(weight, nn.Parameter) or weight.ndim != 4 or weight.shape[-1] != 2:
                raise ValueError('FFT parameters must be materialized before adaptation: ' + name)
            grid = tuple(weight.shape[:3])
            if grid != HISTORICAL_GRID[name]:
                raise ValueError('Unexpected FFT parameter grid for ' + name + ': ' + str(grid))
            grids[name] = grid
        # Initialize all four candidate slots in a stable order even when a stage
        # has no detection FFT yet. No call touches CUDA RNG or shared parameters.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 19007)
            candidates = {
                name: DynamicAdapter(c, h, 2 * (fw - 1))
                for name, (c, h, fw) in ((n, grids[n]) for n in FFT_SLOTS)
            }
        for name, block in blocks.items():
            if block is None:
                continue
            if not isinstance(block.ff, DynamicAdapter):
                weight = block.ff.complex_weight
                block.ff = candidates[name].to(device=weight.device, dtype=weight.dtype)
            dynamic_slots.append(name)
            old_id, old_value = gamma_before[name]
            if id(block.gamma) != old_id or not torch.equal(block.gamma.detach(), old_value):
                raise AssertionError('DFFormer changed a residual gamma: ' + name)

    after = model.state_dict()
    shared = sorted(before_keys & set(after))
    changed = [name for name in shared if not torch.equal(before[name], after[name])]
    if changed:
        raise AssertionError('Variant adaptation changed shared tensors: ' + repr(changed))
    after_ids = {name: id(p) for name, p in model.named_parameters()}
    if any(before_ids[name] != after_ids[name] for name in before_ids.keys() & after_ids.keys()):
        raise AssertionError('Variant adaptation replaced a shared parameter object.')
    metadata = {
        'variant': variant, 'seed': seed, 'implementation_sha256': _sha(__file__),
        'interface': 'in_place_apply_no_checkpoint_load_no_forward_no_shared_reset',
        'parameter_count_before': before_numel,
        'parameter_count_after': sum(p.numel() for p in model.parameters()),
        'removed_keys': sorted(before_keys - set(after)),
        'added_keys': sorted(set(after) - before_keys),
        'shared_tensor_count_preserved': len(shared),
        'strict_sds': variant == 'strict_sds',
        'strict_sds_kernel': 'neighbors=-W; center=sum(W)-W_center' if variant == 'strict_sds' else None,
        'dynamic_slots': dynamic_slots,
        'dynamic_seed': seed + 19007 if variant == 'dfformer' else None,
        'dynamic_vendor_path': str(VENDOR) if variant == 'dfformer' else None,
        'dynamic_vendor_sha256': _sha(VENDOR) if variant == 'dfformer' else None,
        'inheritance_policy': inheritance_policy(variant),
        'modules_present': {name: getattr(model, name, None) is not None for name in
            ('spatial_stem', *FFT_SLOTS, 'serial_p5_context', 'freq_spatial_gate', 'det_freq_attn')},
        'complex_parameter_shapes': {name: list(value.shape) for name, value in after.items()
                                     if 'complex_weight' in name},
    }
    model._staged_variant_identity = identity
    model._staged_variant_metadata = metadata
    return dict(metadata)
