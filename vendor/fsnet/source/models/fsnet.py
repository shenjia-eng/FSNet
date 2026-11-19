# FSNet unifies crown localization and counting in one model.
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher_crowd
from .spatial_diff import SpatialDiffFeature, SpatialDiffStem
from .ff_parser import FFParserBlock
from .amp_phase_ff import AmpPhaseFFTBlock, HighFreqPhaseGuide, CrossDomainHFFusion
from .wavelet_enhance import WaveletEnhanceBlock
from .channel_mamba_refine import ChannelMambaRefine
from .fpn_smooth import FPNSmooth
from .vss_refine import VSSRefine
from .freq_spatial_gate import FreqSpatialGate
from .neck_v2 import CrossScaleFFParser, NeckV2
from .neck_p3c import SpatialTopDownAfterFF
from .parallel_fusion import (
    DetDualFreqAttention,
    FPNConcatFuseMultiScale,
    P5GlobalContext,
    ParallelDetRefineVote,
    ParallelDetTripleVote,
    ParallelFPNVote,
    ParallelStemVotePerLevel,
    SEChannelAttention,
)

import numpy as np
import time

# the network frmawork of the regression branch
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)
    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 2)

# the network frmawork of the classification branch
class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()
    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, _ = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

# generate the reference points in grid layout
def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points
# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points

# this class generate all reference points on all pyramid levels
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2**p, row=self.row, line=self.line)
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        return torch.from_numpy(all_anchor_points.astype(np.float32)).to(device=image.device)

class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(Decoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)


    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]

def _input_tensor(samples):
    if isinstance(samples, NestedTensor):
        return samples.tensors
    return samples


FSNET_PATCH_INPUT_SIDE = 128


def infer_fsnet_input_side(args, training=True):
    """
    Network input side (H=W) before backbone — same rules as SHHA dataloader.
    """
    if training and not getattr(args, 'train_no_patch', False):
        return int(getattr(args, 'patch_side', FSNET_PATCH_INPUT_SIDE) or FSNET_PATCH_INPUT_SIDE)
    fixed = int(getattr(args, 'shha_fixed_side', 0) or 0)
    if fixed > 0:
        return fixed
    if not training:
        infer_side = int(getattr(args, 'infer_fixed_side', 0) or 0)
        if infer_side > 0:
            return infer_side
    return None


def det_feat_spatial_size(input_side, use_high_res_fpn):
    stride = 8 if use_high_res_fpn else 16
    h = max(1, int(input_side) // stride)
    return h, h // 2 + 1


def fpn_scale_spatial_sizes(input_side):
    """P3/P4/P5 spatial (h) and rfft width (w) for VGG-FPN strides 8/16/32."""
    side = max(1, int(input_side))
    scales = {
        'p3': max(1, side // 8),
        'p4': max(1, side // 16),
        'p5': max(1, side // 32),
    }
    return {
        name: (h, h // 2 + 1)
        for name, h in scales.items()
    }


def _make_ffparser_block(h, w, use_amp_phase=False, cutoff=0.4,
                         use_freq_attn=False, attn_reduction=16, num_bands=1,
                         ffparser_strategy='complex', dynamic_num_bands=2,
                         dynamic_mlp_hidden=64, afno_reduction=4):
    if use_amp_phase:
        if h is not None and w is not None:
            return AmpPhaseFFTBlock(dim=256, h=h, w=w, cutoff=cutoff)
        return AmpPhaseFFTBlock(dim=256, cutoff=cutoff)
    kwargs = dict(
        use_freq_attn=use_freq_attn,
        attn_reduction=attn_reduction,
        num_bands=num_bands,
        ffparser_strategy=ffparser_strategy,
        dynamic_num_bands=dynamic_num_bands,
        dynamic_mlp_hidden=dynamic_mlp_hidden,
        afno_reduction=afno_reduction,
    )
    if h is not None and w is not None:
        return FFParserBlock(dim=256, h=h, w=w, **kwargs)
    return FFParserBlock(dim=256, **kwargs)


class FSNet(nn.Module):
    def __init__(
        self,
        backbone,
        row=2,
        line=2,
        anchor_level=3,
        use_spatial_diff=False,
        use_high_res_fpn=False,
        use_ffparser=False,
        ffparser_mode='det',
        ffparser_det_hw=None,
        ffparser_fpn_hw=None,
        use_wavelet=False,
        use_channel_mamba=False,
        use_fpn_smooth=False,
        use_vssblock=False,
        vssblock_depth=1,
        vmamba_cfg=None,
        s3mamba_root=None,
        fusion_mode='serial',
        fusion_vote='softmax',
        use_amp_phase=False,
        amp_phase_cutoff=0.4,       # 高频/低频分界点
        use_freq_spatial_gate=False,
        gate_position='after_det_ff',  # 'before_det_ff' | 'after_det_ff'
        gate_mode='mul',              # 'mul' | 'add'
        use_neck_se=False,            # serial: SE on fused det (before det-FF)
        use_p5_context=False,         # serial: GAP(P5) broadcast add to det
        use_freq_attn=False,          # serial: FreqAmpChannelAttention after FFParser
        attn_reduction=16,            # reduction ratio for FreqAmpChannelAttention
        use_det_freq_attn=False,
        det_freq_attn_reduction=16,
        det_freq_attn_identity_init=True,
        det_freq_attn_position='after_det_ff',  # 'after_det_ff' | 'after_gate'
        use_det_dual_freq_attn=False,
        det_dual_freq_beta=0.1,
        ffparser_num_bands=1,               # 1=full-band FFParser (default)
        ffparser_strategy='complex',
        dynamic_num_bands=2,                # DFFormer-style dynamic band count
        dynamic_mlp_hidden=64,              # hidden width for dynamic filter generator
        afno_reduction=4,                   # channel reduction for AFNO-style spectral MLP
        spatial_stem_type='laplacian',      # laplacian|plain|sobel|scharr|dog|log|fixed_laplacian
        stem_kernel_size=7,                  # SDS stem conv kernel size (3 or 7)
        stem_freeze_diff=False,              # freeze diff.* weights
        p5_pool_type='avg',                 # 'avg' | 'max'
        det_fuse_levels='p3p4',             # 'p3p4' | 'p3p4p5'
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        self.use_spatial_diff = use_spatial_diff
        self.use_high_res_fpn = use_high_res_fpn
        self.use_ffparser = use_ffparser
        self.ffparser_mode = ffparser_mode if use_ffparser else 'none'
        self.use_wavelet = use_wavelet
        self.use_channel_mamba = use_channel_mamba
        self.use_fpn_smooth = use_fpn_smooth
        self.use_vssblock = use_vssblock
        if fusion_mode not in ('serial', 'parallel', 'v2', 'crossscale', 'native_spatial'):
            raise ValueError(
                "fusion_mode must be 'serial', 'parallel', 'v2', 'crossscale', or "
                "'native_spatial', got %r" % fusion_mode
            )
        if fusion_vote not in ('sigmoid', 'softmax', 'concat'):
            raise ValueError("fusion_vote must be 'sigmoid', 'softmax', or 'concat', got %r" % fusion_vote)
        self.fusion_mode = fusion_mode
        self.fusion_vote = fusion_vote
        self.use_amp_phase = use_amp_phase
        self.amp_phase_cutoff = amp_phase_cutoff
        self.use_freq_spatial_gate = use_freq_spatial_gate
        if gate_position not in ('before_det_ff', 'after_det_ff'):
            raise ValueError(
                "gate_position must be 'before_det_ff' or 'after_det_ff', got %r"
                % gate_position
            )
        self.gate_position = gate_position
        if gate_mode not in ('mul', 'add'):
            raise ValueError("gate_mode must be 'mul' or 'add', got %r" % gate_mode)
        self.gate_mode = gate_mode
        self.use_neck_se = use_neck_se
        self.use_p5_context = use_p5_context
        self.use_freq_attn = use_freq_attn
        self.attn_reduction = attn_reduction
        self.use_det_freq_attn = use_det_freq_attn
        self.det_freq_attn_reduction = det_freq_attn_reduction
        if det_freq_attn_position not in ('after_det_ff', 'after_gate'):
            raise ValueError(
                "det_freq_attn_position must be 'after_det_ff' or 'after_gate', got %r"
                % det_freq_attn_position
            )
        self.det_freq_attn_position = det_freq_attn_position
        self.use_det_dual_freq_attn = use_det_dual_freq_attn
        self.det_dual_freq_beta = det_dual_freq_beta
        if det_fuse_levels not in ('p3p4', 'p3p4p5'):
            raise ValueError(
                "det_fuse_levels must be 'p3p4' or 'p3p4p5', got %r" % det_fuse_levels
            )
        self.det_fuse_levels = det_fuse_levels
        self.ffparser_num_bands = int(ffparser_num_bands)
        self.ffparser_strategy = ffparser_strategy
        self.dynamic_num_bands = int(dynamic_num_bands)
        self.dynamic_mlp_hidden = int(dynamic_mlp_hidden)
        self.afno_reduction = int(afno_reduction)
        self.spatial_stem_type = spatial_stem_type
        self.stem_kernel_size = stem_kernel_size
        self.stem_freeze_diff = stem_freeze_diff
        self.p5_pool_type = p5_pool_type
        self.cross_scale_ff = None
        self.neck_v2 = None
        self.spatial_topdown = None
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(num_features_in=256, \
                                            num_classes=self.num_classes, \
                                            num_anchor_points=num_anchor_points)

        self.anchor_points = AnchorPoints(pyramid_levels=[anchor_level], row=row, line=line)

        self.fpn = Decoder(256, 512, 512)
        if use_high_res_fpn:
            fuse_in = 768 if det_fuse_levels == 'p3p4p5' else 512
            self.fpn_fuse = nn.Sequential(
                nn.Conv2d(fuse_in, 256, kernel_size=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        else:
            self.fpn_fuse = None

        if use_spatial_diff:
            self.spatial_stem = SpatialDiffStem(
                in_channels=3,
                mid_channels=32,
                kernel_size=stem_kernel_size,
                stem_type=spatial_stem_type,
                freeze_diff=stem_freeze_diff,
            )
        else:
            self.spatial_stem = None

        self.ffparser = None
        self.ffparser_p3 = None
        self.ffparser_p4 = None
        self.ffparser_p5 = None
        if fusion_mode == 'v2':
            fpn_hw = ffparser_fpn_hw or {}
            p3_hw = fpn_hw.get('p3')
            self.cross_scale_ff = CrossScaleFFParser(
                256,
                p3_ff_hw=p3_hw,
                use_freq_attn=use_freq_attn,
                attn_reduction=attn_reduction,
            )
            self.neck_v2 = NeckV2(
                channels=256,
                use_high_res_fpn=use_high_res_fpn,
                det_ff_hw=ffparser_det_hw,
                use_freq_spatial_gate=use_freq_spatial_gate,
                use_freq_attn=use_freq_attn,
                attn_reduction=attn_reduction,
            )
            self.use_ffparser = True
            self.ffparser_mode = 'v2'
        elif fusion_mode == 'crossscale':
            fpn_hw = ffparser_fpn_hw or {}
            p3_hw = fpn_hw.get('p3')
            self.cross_scale_ff = CrossScaleFFParser(
                256,
                p3_ff_hw=p3_hw,
                use_freq_attn=use_freq_attn,
                attn_reduction=attn_reduction,
            )
            if ffparser_mode in ('det', 'both'):
                if ffparser_det_hw is not None:
                    fh, fw = ffparser_det_hw
                    self.ffparser = _make_ffparser_block(fh, fw, use_amp_phase, amp_phase_cutoff,
                                                         use_freq_attn=use_freq_attn,
                                                         attn_reduction=attn_reduction, num_bands=ffparser_num_bands,
                                                         ffparser_strategy=ffparser_strategy,
                                                         dynamic_num_bands=dynamic_num_bands,
                                                         dynamic_mlp_hidden=dynamic_mlp_hidden,
                                                         afno_reduction=afno_reduction)
                else:
                    self.ffparser = _make_ffparser_block(None, None, use_amp_phase, amp_phase_cutoff,
                                                         use_freq_attn=use_freq_attn,
                                                         attn_reduction=attn_reduction, num_bands=ffparser_num_bands,
                                                         ffparser_strategy=ffparser_strategy,
                                                         dynamic_num_bands=dynamic_num_bands,
                                                         dynamic_mlp_hidden=dynamic_mlp_hidden,
                                                         afno_reduction=afno_reduction)
            self.use_ffparser = True
            self.ffparser_mode = ffparser_mode
        elif fusion_mode == 'native_spatial':
            fpn_hw = ffparser_fpn_hw or {}
            if ffparser_mode in ('det', 'both'):
                if ffparser_det_hw is not None:
                    fh, fw = ffparser_det_hw
                    self.ffparser = _make_ffparser_block(fh, fw, use_amp_phase, amp_phase_cutoff,
                                                         use_freq_attn=use_freq_attn,
                                                         attn_reduction=attn_reduction, num_bands=ffparser_num_bands,
                                                         ffparser_strategy=ffparser_strategy,
                                                         dynamic_num_bands=dynamic_num_bands,
                                                         dynamic_mlp_hidden=dynamic_mlp_hidden,
                                                         afno_reduction=afno_reduction)
                else:
                    self.ffparser = _make_ffparser_block(None, None, use_amp_phase, amp_phase_cutoff,
                                                         use_freq_attn=use_freq_attn,
                                                         attn_reduction=attn_reduction, num_bands=ffparser_num_bands,
                                                         ffparser_strategy=ffparser_strategy)
            if ffparser_mode in ('fpn', 'both'):
                p3_hw = fpn_hw.get('p3')
                p4_hw = fpn_hw.get('p4')
                p5_hw = fpn_hw.get('p5')
                self.ffparser_p3 = _make_ffparser_block(
                    p3_hw[0] if p3_hw else None,
                    p3_hw[1] if p3_hw else None,
                    use_amp_phase, amp_phase_cutoff,
                    use_freq_attn=use_freq_attn,
                    attn_reduction=attn_reduction,
                    ffparser_strategy=ffparser_strategy,
                    dynamic_num_bands=dynamic_num_bands,
                    dynamic_mlp_hidden=dynamic_mlp_hidden,
                    afno_reduction=afno_reduction,
                )
                self.ffparser_p4 = _make_ffparser_block(
                    p4_hw[0] if p4_hw else None,
                    p4_hw[1] if p4_hw else None,
                    use_amp_phase, amp_phase_cutoff,
                    use_freq_attn=use_freq_attn,
                    attn_reduction=attn_reduction,
                    ffparser_strategy=ffparser_strategy,
                )
                self.ffparser_p5 = _make_ffparser_block(
                    p5_hw[0] if p5_hw else None,
                    p5_hw[1] if p5_hw else None,
                    use_amp_phase, amp_phase_cutoff,
                    use_freq_attn=use_freq_attn,
                    attn_reduction=attn_reduction,
                    ffparser_strategy=ffparser_strategy,
                )
            self.spatial_topdown = SpatialTopDownAfterFF(channels=256)
            self.use_ffparser = True
            self.ffparser_mode = ffparser_mode
        elif use_ffparser:
            fpn_hw = ffparser_fpn_hw or {}
            if ffparser_mode in ('det', 'both'):
                if ffparser_det_hw is not None:
                    fh, fw = ffparser_det_hw
                    self.ffparser = _make_ffparser_block(fh, fw, use_amp_phase, amp_phase_cutoff,
                                                         use_freq_attn=use_freq_attn,
                                                         attn_reduction=attn_reduction, num_bands=ffparser_num_bands,
                                                         ffparser_strategy=ffparser_strategy)
                else:
                    self.ffparser = _make_ffparser_block(None, None, use_amp_phase, amp_phase_cutoff,
                                                         use_freq_attn=use_freq_attn,
                                                         attn_reduction=attn_reduction, num_bands=ffparser_num_bands,
                                                         ffparser_strategy=ffparser_strategy)
            if ffparser_mode in ('fpn', 'both'):
                p3_hw = fpn_hw.get('p3')
                p4_hw = fpn_hw.get('p4')
                p5_hw = fpn_hw.get('p5')
                self.ffparser_p3 = _make_ffparser_block(
                    p3_hw[0] if p3_hw else None,
                    p3_hw[1] if p3_hw else None,
                    use_amp_phase, amp_phase_cutoff,
                    use_freq_attn=use_freq_attn,
                    attn_reduction=attn_reduction,
                    ffparser_strategy=ffparser_strategy,
                )
                self.ffparser_p4 = _make_ffparser_block(
                    p4_hw[0] if p4_hw else None,
                    p4_hw[1] if p4_hw else None,
                    use_amp_phase, amp_phase_cutoff,
                    use_freq_attn=use_freq_attn,
                    attn_reduction=attn_reduction,
                    ffparser_strategy=ffparser_strategy,
                )
                self.ffparser_p5 = _make_ffparser_block(
                    p5_hw[0] if p5_hw else None,
                    p5_hw[1] if p5_hw else None,
                    use_amp_phase, amp_phase_cutoff,
                    use_freq_attn=use_freq_attn,
                    attn_reduction=attn_reduction,
                    ffparser_strategy=ffparser_strategy,
                )
            if fusion_mode == 'parallel' and self.ffparser is None:
                if ffparser_det_hw is not None:
                    fh, fw = ffparser_det_hw
                    self.ffparser = _make_ffparser_block(fh, fw, use_amp_phase, amp_phase_cutoff,
                                                         use_freq_attn=use_freq_attn,
                                                         attn_reduction=attn_reduction, num_bands=ffparser_num_bands,
                                                         ffparser_strategy=ffparser_strategy)
                else:
                    self.ffparser = _make_ffparser_block(None, None, use_amp_phase, amp_phase_cutoff,
                                                         use_freq_attn=use_freq_attn,
                                                         attn_reduction=attn_reduction, num_bands=ffparser_num_bands,
                                                         ffparser_strategy=ffparser_strategy)

        self.fpn_smooth = FPNSmooth(channels=256) if use_fpn_smooth else None
        self.wavelet_enhance = WaveletEnhanceBlock(in_channels=256) if use_wavelet else None
        self.channel_mamba = ChannelMambaRefine(in_channels=256) if use_channel_mamba else None
        self.vss_refine = None
        if use_vssblock:
            self.vss_refine = VSSRefine(
                hidden_dim=256,
                depth=vssblock_depth,
                vmamba_cfg=vmamba_cfg,
                s3mamba_root=s3mamba_root,
            )

        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        self.count_head_ms = None

        self.stem_vote = None
        self.spatial_feat_diff = None
        self.fpn_vote = None
        self.fpn_l2_fuse = None
        self.ca_p3 = None
        self.ca_p4 = None
        self.p5_context = None
        self.det_triple_vote = None
        self.det_refine_vote = None
        if fusion_mode == 'parallel':
            if use_spatial_diff:
                self.stem_vote = ParallelStemVotePerLevel(fusion_vote=fusion_vote)
                self.spatial_feat_diff = nn.ModuleDict({
                    str(k): SpatialDiffFeature(c)
                    for k, c in ParallelStemVotePerLevel.VGG16_BN_LEVEL_CHANNELS.items()
                })
            use_ffpn = use_ffparser and ffparser_mode in ('fpn', 'both')
            if use_ffpn:
                self.fpn_l2_fuse = FPNConcatFuseMultiScale(channels=256)
                self.ca_p3 = SEChannelAttention(channels=256, identity_init=False)
                self.ca_p4 = SEChannelAttention(channels=256, identity_init=False)
                self.p5_context = P5GlobalContext(channels=256, pool_type=p5_pool_type)
                self.count_head_ms = nn.Sequential(
                    nn.Linear(256 * 3, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 1),
                    nn.Softplus(),
                )
            elif use_fpn_smooth:
                self.fpn_vote = ParallelFPNVote(
                    use_smooth=True,
                    use_ffpn=False,
                    fusion_vote=fusion_vote,
                )
            if use_ffpn or (use_ffparser and ffparser_mode in ('fpn', 'both')):
                self.det_triple_vote = ParallelDetTripleVote(
                    use_path_f=use_ffpn,
                    use_path_d=False,
                    fusion_vote=fusion_vote,
                )
            use_ffdet = use_ffparser and ffparser_mode in ('det', 'both')
            if fusion_mode == 'parallel':
                use_ffdet = use_ffparser
            if use_ffdet or use_channel_mamba or use_wavelet or use_vssblock:
                self.det_refine_vote = ParallelDetRefineVote(
                    use_ffparser=use_ffdet,
                    use_channel_mamba=use_channel_mamba,
                    use_wavelet=use_wavelet,
                    use_vss=use_vssblock,
                    fusion_vote=fusion_vote,
                )

        self.hf_phase_guide = None
        if use_amp_phase and fusion_mode == 'parallel':
            det_h, det_w = (None, None)
            if ffparser_det_hw is not None:
                det_h, det_w = ffparser_det_hw
            self.hf_phase_guide = HighFreqPhaseGuide(
                channels=256, det_h=det_h, det_w=det_w,
                cutoff=amp_phase_cutoff,
            )

        self.freq_spatial_gate = None
        if use_freq_spatial_gate and fusion_mode == 'serial':
            self.freq_spatial_gate = FreqSpatialGate(channels=256, mode=gate_mode)

        self.serial_neck_se = None
        self.serial_p5_context = None
        self.det_freq_attn = None
        if fusion_mode == 'serial':
            if use_neck_se:
                self.serial_neck_se = SEChannelAttention(channels=256, identity_init=False)
            if use_p5_context:
                self.serial_p5_context = P5GlobalContext(channels=256, pool_type=p5_pool_type)
            if use_det_dual_freq_attn:
                self.det_freq_attn = DetDualFreqAttention(
                    channels=256,
                    reduction=det_freq_attn_reduction,
                    beta_init=det_dual_freq_beta,
                    identity_init=det_freq_attn_identity_init,
                )
            elif use_det_freq_attn:
                self.det_freq_attn = SEChannelAttention(
                    channels=256,
                    reduction=det_freq_attn_reduction,
                    identity_init=det_freq_attn_identity_init,
                    identity_bias=3.0,
                )

    def _run_backbone(self, images):
        backbone_out = self.backbone(images)
        if isinstance(backbone_out, tuple):
            features, backbone_aux = backbone_out
        else:
            features, backbone_aux = backbone_out, {}
        return features, backbone_aux

    def _forward_backbone(self, images):
        if (
            self.fusion_mode == 'parallel'
            and self.stem_vote is not None
            and self.spatial_feat_diff is not None
        ):
            features, backbone_aux = self._run_backbone(images)
            merged = list(features)
            for idx in (1, 2):
                diff_feat = self.spatial_feat_diff[str(idx)](features[idx])
                merged[idx] = self.stem_vote.merge_level(features[idx], diff_feat, idx)
            return merged, backbone_aux

        stem_out = None
        if self.spatial_stem is not None and self.fusion_mode != 'parallel':
            images = self.spatial_stem(images)
            stem_out = images
        features, backbone_aux = self._run_backbone(images)
        if stem_out is not None:
            backbone_aux['spatial_stem_out'] = stem_out
        return features, backbone_aux

    def _apply_parallel_fpn_fuse(self, features_fpn):
        p3, p4, p5 = features_fpn
        if self.fpn_l2_fuse is not None:
            p3 = self.fpn_l2_fuse.forward_scale('p3', p3, self.ffparser_p3)
            p4 = self.fpn_l2_fuse.forward_scale('p4', p4, self.ffparser_p4)
            p5 = self.fpn_l2_fuse.forward_scale('p5', p5, self.ffparser_p5)
            return [p3, p4, p5]

        vote = self.fpn_vote
        if vote is None:
            return features_fpn
        smooth = self.fpn_smooth
        p3 = vote(p3, smooth.smooth_p3 if smooth else None, self.ffparser_p3)
        p4 = vote(p4, smooth.smooth_p4 if smooth else None, self.ffparser_p4)
        p5 = vote(p5, smooth.smooth_p5 if smooth else None, self.ffparser_p5)
        return [p3, p4, p5]

    def _apply_fpn_ffparser(self, features_fpn):
        p3, p4, p5 = features_fpn
        if self.ffparser_p3 is not None:
            p3 = self.ffparser_p3(p3)
        if self.ffparser_p4 is not None:
            p4 = self.ffparser_p4(p4)
        if self.ffparser_p5 is not None:
            p5 = self.ffparser_p5(p5)
        return [p3, p4, p5]

    def _fuse_p3_p4(self, p3, p4):
        if p4.shape[-2:] != p3.shape[-2:]:
            p4 = F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        return self.fpn_fuse(torch.cat([p3, p4], dim=1))

    def _fuse_p3_p4_p5(self, p3, p4, p5):
        target = p3.shape[-2:]
        if p4.shape[-2:] != target:
            p4 = F.interpolate(p4, size=target, mode='nearest')
        if p5.shape[-2:] != target:
            p5 = F.interpolate(p5, size=target, mode='nearest')
        return self.fpn_fuse(torch.cat([p3, p4, p5], dim=1))

    def _apply_triple_det_vote(self, p3_f, p4_f, p5_f=None):
        """Path-R: Fuse(P_f); Path-F: Fuse(CA(P_f)); both + P5 global context."""
        ctx = None
        if p5_f is not None and self.p5_context is not None:
            ctx = self.p5_context(p5_f, p3_f.shape[-2:])

        if not self.use_high_res_fpn:
            det_r = p4_f
            det_f = self.ca_p4(p4_f) if self.ca_p4 is not None else None
        else:
            det_r = self._fuse_p3_p4(p3_f, p4_f)
            det_f = None
            if self.ca_p3 is not None and self.ca_p4 is not None:
                det_f = self._fuse_p3_p4(self.ca_p3(p3_f), self.ca_p4(p4_f))

        if ctx is not None:
            det_r = det_r + ctx
            if det_f is not None:
                det_f = det_f + ctx

        return self.det_triple_vote(det_r, det_f, None)

    def _predict_count(self, det_feat, features_fpn):
        if self.count_head_ms is not None:
            p3, p4, p5 = features_fpn
            pooled = torch.cat([
                F.adaptive_avg_pool2d(p3, 1).flatten(1),
                F.adaptive_avg_pool2d(p4, 1).flatten(1),
                F.adaptive_avg_pool2d(p5, 1).flatten(1),
            ], dim=1)
            return self.count_head_ms(pooled).squeeze(-1)
        return self.count_head(det_feat).squeeze(-1)

    def _apply_parallel_det_refine(self, det_feat):
        return self.det_refine_vote(
            det_feat,
            ff_fn=self.ffparser,
            mamba_fn=self.channel_mamba,
            wave_fn=self.wavelet_enhance,
            vss_fn=self.vss_refine,
        )

    def _fusion_vote_dict(self):
        votes = {}
        if self.stem_vote is not None:
            votes.update(self.stem_vote.vote_dict())
        if self.fpn_vote is not None:
            votes.update(self.fpn_vote.vote_dict())
        if self.det_triple_vote is not None:
            votes.update(self.det_triple_vote.vote_dict())
        if self.det_refine_vote is not None:
            votes.update(self.det_refine_vote.vote_dict())
        return votes

    def _detection_features(self, features_fpn):
        if not self.use_high_res_fpn:
            return features_fpn[1]
        p3, p4, p5 = features_fpn[0], features_fpn[1], features_fpn[2]
        if self.det_fuse_levels == 'p3p4p5':
            return self._fuse_p3_p4_p5(p3, p4, p5)
        return self._fuse_p3_p4(p3, p4)

    def _apply_serial_neck_extras(self, det_feat, p5):
        """Serial neck: P5 global context + SE on fused det (before det-FF)."""
        if self.serial_p5_context is not None:
            det_feat = det_feat + self.serial_p5_context(p5, det_feat.shape[-2:])
        if self.serial_neck_se is not None:
            det_feat = self.serial_neck_se(det_feat)
        return det_feat

    def _apply_p3_freq_spatial_gate(self, det_feat, p3_ref):
        if self.freq_spatial_gate is None:
            return det_feat
        if p3_ref.shape[-2:] != det_feat.shape[-2:]:
            p3_ref = F.interpolate(
                p3_ref, size=det_feat.shape[-2:], mode='bilinear', align_corners=False
            )
        return self.freq_spatial_gate(det_feat, p3_ref)

    def _apply_serial_det_head(self, features_fpn, p5_fused, debug_vis=None):
        det_feat = self._detection_features(features_fpn)
        det_feat = self._apply_serial_neck_extras(det_feat, p5_fused)
        p3_ref = features_fpn[0]
        if self.freq_spatial_gate is not None and self.gate_position == 'before_det_ff':
            det_feat = self._apply_p3_freq_spatial_gate(det_feat, p3_ref)
            if debug_vis is not None:
                debug_vis['det_after_gate'] = det_feat.detach()
                if getattr(self.freq_spatial_gate, 'last_gate', None) is not None:
                    debug_vis['freq_gate_map'] = self.freq_spatial_gate.last_gate.detach()
        if self.ffparser is not None:
            det_feat = self.ffparser(det_feat)
            if debug_vis is not None:
                debug_vis['det_after_ffparser'] = det_feat.detach()
        if (
            self.det_freq_attn is not None
            and self.det_freq_attn_position == 'after_det_ff'
        ):
            det_feat = self.det_freq_attn(det_feat)
            if debug_vis is not None:
                debug_vis['det_after_se'] = det_feat.detach()
                if getattr(self.det_freq_attn, 'last_weights', None) is not None:
                    debug_vis['det_se_weights'] = self.det_freq_attn.last_weights.detach()
        if self.freq_spatial_gate is not None and self.gate_position == 'after_det_ff':
            det_feat = self._apply_p3_freq_spatial_gate(det_feat, p3_ref)
            if debug_vis is not None:
                debug_vis['det_after_gate'] = det_feat.detach()
                if getattr(self.freq_spatial_gate, 'last_gate', None) is not None:
                    debug_vis['freq_gate_map'] = self.freq_spatial_gate.last_gate.detach()
        if (
            self.det_freq_attn is not None
            and self.det_freq_attn_position == 'after_gate'
        ):
            det_feat = self.det_freq_attn(det_feat)
            if debug_vis is not None:
                debug_vis['det_after_se'] = det_feat.detach()
                if getattr(self.det_freq_attn, 'last_weights', None) is not None:
                    debug_vis['det_se_weights'] = self.det_freq_attn.last_weights.detach()
        if self.wavelet_enhance is not None:
            det_feat = self.wavelet_enhance(det_feat)
        if self.channel_mamba is not None:
            det_feat = self.channel_mamba(det_feat)
        if self.vss_refine is not None:
            det_feat = self.vss_refine(det_feat)
        return det_feat

    def forward(self, samples: NestedTensor):
        images = _input_tensor(samples)
        debug_vis = {'input_image': images.detach()}

        features, backbone_aux = self._forward_backbone(images)
        if 'spatial_stem_out' in backbone_aux:
            debug_vis['spatial_stem_out'] = backbone_aux['spatial_stem_out'].detach()

        # forward the feature pyramid
        features_fpn = self.fpn([features[1], features[2], features[3]])

        if self.fusion_mode == 'v2':
            if self.fpn_smooth is not None:
                features_fpn = self.fpn_smooth(features_fpn)
            p3_fused, p4_fused, p5_fused = features_fpn
            p3_fused, p4_fused, p5_fused = self.cross_scale_ff(
                p3_fused, p4_fused, p5_fused
            )
            features_fpn = [p3_fused, p4_fused, p5_fused]
            det_feat = self.neck_v2(p3_fused, p4_fused, p5_fused, self.fpn_fuse)
        elif self.fusion_mode == 'crossscale':
            if self.fpn_smooth is not None:
                features_fpn = self.fpn_smooth(features_fpn)
            p3_fused, p4_fused, p5_fused = features_fpn
            p3_fused, p4_fused, p5_fused = self.cross_scale_ff(
                p3_fused, p4_fused, p5_fused
            )
            features_fpn = [p3_fused, p4_fused, p5_fused]
            det_feat = self._apply_serial_det_head(features_fpn, p5_fused)
        elif self.fusion_mode == 'native_spatial':
            if self.fpn_smooth is not None:
                features_fpn = self.fpn_smooth(features_fpn)
            features_fpn = self._apply_fpn_ffparser(features_fpn)
            p3_fused, p4_fused, p5_fused = self.spatial_topdown(*features_fpn)
            features_fpn = [p3_fused, p4_fused, p5_fused]
            det_feat = self._apply_serial_det_head(features_fpn, p5_fused)
        elif self.fusion_mode == 'parallel' and self.fpn_l2_fuse is not None:
            features_fpn = self._apply_parallel_fpn_fuse(features_fpn)
        elif self.fusion_mode == 'parallel' and self.fpn_vote is not None:
            features_fpn = self._apply_parallel_fpn_fuse(features_fpn)
        elif self.fpn_smooth is not None:
            features_fpn = self.fpn_smooth(features_fpn)

        if self.fusion_mode not in ('v2', 'crossscale', 'native_spatial'):
            p3_fused, p4_fused, p5_fused = features_fpn[0], features_fpn[1], features_fpn[2]

        if self.fusion_mode == 'parallel' and self.det_triple_vote is not None:
            det_feat = self._apply_triple_det_vote(p3_fused, p4_fused, p5_fused)
        elif self.fusion_mode == 'parallel':
            det_feat = self._detection_features(features_fpn)
        elif self.fusion_mode not in ('v2', 'crossscale', 'native_spatial'):
            if self.ffparser_p3 is not None:
                features_fpn = self._apply_fpn_ffparser(features_fpn)
            debug_vis['p3_fpn'] = features_fpn[0].detach()
            debug_vis['p4_fpn'] = features_fpn[1].detach()
            debug_vis['p5_fpn'] = features_fpn[2].detach()
            det_before = self._detection_features(features_fpn)
            debug_vis['det_after_fuse'] = det_before.detach()
            det_feat = self._apply_serial_det_head(
                features_fpn, features_fpn[2], debug_vis=debug_vis
            )
            p3_fused, p4_fused, p5_fused = (
                features_fpn[0], features_fpn[1], features_fpn[2]
            )

        if self.fusion_mode in ('v2', 'crossscale', 'native_spatial'):
            pass
        elif self.fusion_mode == 'parallel' and self.det_refine_vote is not None:
            det_feat = self._apply_parallel_det_refine(det_feat)
        elif self.fusion_mode == 'parallel' and self.ffparser is not None:
            det_feat = self.ffparser(det_feat)

        if self.fusion_mode == 'parallel' and self.hf_phase_guide is not None:
            det_feat = self.hf_phase_guide(det_feat, spatial_hf=p3_fused)

        batch_size = features[0].shape[0]
        # Dense point regression: offset scaled ×100 to match anchor coordinate range.
        regression = self.regression(det_feat) * 100
        classification = self.classification(det_feat)
        anchor_points = self.anchor_points(images).repeat(batch_size, 1, 1)
        # decode the points as prediction
        output_coord = regression + anchor_points
        output_class = classification
        output_count = self._predict_count(det_feat, features_fpn)
        class_prob = torch.nn.functional.softmax(output_class, -1)[..., 1]
        b, n = class_prob.shape
        side = int(n ** 0.5)
        if side * side == n:
            debug_vis['class_score_map'] = class_prob.view(b, 1, side, side).detach()
        self._debug_last = debug_vis
        out = {'pred_logits': output_class, 'pred_points': output_coord, 'pred_count_aux': output_count}
        if 'competition_penalty' in backbone_aux:
            out['loss_competition_aux'] = backbone_aux['competition_penalty']
        if 'fusion_alpha' in backbone_aux:
            out['fusion_alpha'] = backbone_aux['fusion_alpha']
        if self.fusion_mode == 'parallel':
            out['fusion_votes'] = self._fusion_vote_dict()
            out['fusion_vote_mode'] = self.fusion_vote

        return out

class SetCriterion_Crowd(nn.Module):

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        geo_k=3,
        density_adaptive_loss=False,
        density_ref_count=80.0,
        density_gamma=0.5,
        density_weight_min=0.6,
        density_weight_max=1.8,
        density_adaptive_mode='all',
        density_high_thresh=150.0,
        uncertainty_weighting=False,
    ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.geo_k = max(1, int(geo_k))
        self.density_adaptive_loss = density_adaptive_loss
        self.density_ref_count = float(density_ref_count)
        self.density_gamma = float(density_gamma)
        self.density_weight_min = float(density_weight_min)
        self.density_weight_max = float(density_weight_max)
        self.density_adaptive_mode = density_adaptive_mode
        self.density_high_thresh = float(density_high_thresh)
        self.uncertainty_weighting = bool(uncertainty_weighting)
        if self.uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                "loss_point": nn.Parameter(torch.zeros(1)),
                "loss_geo": nn.Parameter(torch.zeros(1)),
                "loss_count": nn.Parameter(torch.zeros(1)),
            })
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.reshape(-1, src_logits.shape[-1]), target_classes.reshape(-1), self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_points(self, outputs, targets, indices, num_points):

        assert 'pred_points' in outputs
        if not self.density_adaptive_loss:
            idx = self._get_src_permutation_idx(indices)
            src_points = outputs['pred_points'][idx]
            target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            loss_bbox = F.mse_loss(src_points, target_points, reduction='none')
            losses = {}
            losses['loss_point'] = loss_bbox.sum() / num_points
            return losses

        total_loss = outputs['pred_points'].new_zeros(())
        total_weight = outputs['pred_points'].new_zeros(())
        for b_idx, (src_ids, tgt_ids) in enumerate(indices):
            if len(src_ids) == 0:
                continue
            src_points = outputs['pred_points'][b_idx, src_ids]
            target_points = targets[b_idx]['point'][tgt_ids]
            loss_bbox = F.mse_loss(src_points, target_points, reduction='none').sum()
            gt_count = float(targets[b_idx]['point'].shape[0])
            if self.density_adaptive_mode == 'high_only':
                if gt_count < self.density_high_thresh:
                    weight = 1.0
                else:
                    ratio = max(gt_count, 1.0) / max(self.density_ref_count, 1e-6)
                    weight = ratio ** self.density_gamma
                    weight = max(1.0, min(self.density_weight_max, weight))
            else:
                ratio = max(gt_count, 1.0) / max(self.density_ref_count, 1e-6)
                weight = ratio ** self.density_gamma
                weight = max(self.density_weight_min, min(self.density_weight_max, weight))
            total_loss = total_loss + loss_bbox * weight
            total_weight = total_weight + src_points.new_tensor(weight * max(len(src_ids), 1))

        losses = {}
        if total_weight.item() == 0:
            losses['loss_point'] = outputs['pred_points'].new_zeros(())
        else:
            losses['loss_point'] = total_loss / total_weight

        return losses

    def _knn_distance_vector(self, points: torch.Tensor):
        if points.shape[0] < 2:
            return None
        dist = torch.cdist(points, points, p=2)
        diag_mask = torch.eye(dist.shape[0], dtype=torch.bool, device=dist.device)
        dist = dist.masked_fill(diag_mask, float('inf'))
        k_eff = min(self.geo_k, points.shape[0] - 1)
        if k_eff <= 0:
            return None
        knn_d = torch.topk(dist, k=k_eff, largest=False, dim=1).values
        return knn_d.reshape(-1)

    def loss_geometry(self, outputs, targets, indices, num_points):
        del num_points
        pred_points = outputs['pred_points']
        device = pred_points.device
        total = pred_points.new_zeros(())
        valid = 0
        for b_idx, (src_ids, tgt_ids) in enumerate(indices):
            if len(src_ids) < 2:
                continue
            src_pts = pred_points[b_idx, src_ids]
            tgt_pts = targets[b_idx]['point'][tgt_ids]
            src_v = self._knn_distance_vector(src_pts)
            tgt_v = self._knn_distance_vector(tgt_pts)
            if src_v is None or tgt_v is None:
                continue
            # Compare normalized local geometry to avoid scale bias.
            src_v = src_v / (src_v.mean().detach() + 1e-6)
            tgt_v = tgt_v / (tgt_v.mean().detach() + 1e-6)
            total = total + F.l1_loss(src_v, tgt_v)
            valid += 1
        if valid == 0:
            return {'loss_geo': torch.zeros((), device=device)}
        return {'loss_geo': total / valid}

    def loss_count(self, outputs, targets, indices, num_points):
        del indices, num_points
        if 'pred_count_aux' not in outputs:
            return {'loss_count': outputs['pred_points'].new_zeros(())}
        pred_count = outputs['pred_count_aux']
        gt_count = torch.as_tensor(
            [float(t['labels'].shape[0]) for t in targets],
            dtype=pred_count.dtype,
            device=pred_count.device,
        )
        return {'loss_count': F.smooth_l1_loss(pred_count, gt_count)}

    def loss_competition(self, outputs, targets, indices, num_points):
        del targets, indices, num_points
        if 'loss_competition_aux' not in outputs:
            return {'loss_competition': outputs['pred_points'].new_zeros(())}
        return {'loss_competition': outputs['loss_competition_aux']}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
            'geometry': self.loss_geometry,
            'count': self.loss_count,
            'competition': self.loss_competition,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}

        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            if loss == 'competition':
                losses.update(self.get_loss(loss, outputs, targets, indices1, num_boxes))
            elif loss == 'count':
                losses.update(self.get_loss(loss, outputs, targets, indices1, num_boxes))
            else:
                losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))

        if self.uncertainty_weighting:
            # L_i' = exp(-s_i) * L_i + s_i ; s_i is learnable log-variance.
            # Keep CE on fixed weighting; only rebalance regression-style auxiliary losses.
            for name, val in list(losses.items()):
                if name in self.log_vars:
                    s = self.log_vars[name]
                    losses[name] = torch.exp(-s) * val + s

        return losses

def build(args, training):
    num_classes = 1

    backbone = build_backbone(args)
    backbone_name = getattr(args, 'backbone', 'vgg16_bn')
    use_spatial_diff = getattr(args, 'use_spatial_diff', True)
    use_high_res_fpn = getattr(args, 'use_high_res_fpn', True)
    use_ffparser = getattr(args, 'use_ffparser', False)
    use_wavelet = getattr(args, 'use_wavelet', False)
    use_channel_mamba = getattr(args, 'use_channel_mamba', False)
    use_fpn_smooth = getattr(args, 'use_fpn_smooth', False)
    use_vssblock = getattr(args, 'use_vssblock', False)
    vssblock_depth = int(getattr(args, 'vssblock_depth', 1))
    fusion_mode = getattr(args, 'fusion_mode', 'serial')
    fusion_vote = getattr(args, 'fusion_vote', 'softmax')
    use_amp_phase = getattr(args, 'use_amp_phase', False)
    amp_phase_cutoff = getattr(args, 'amp_phase_cutoff', 0.4)
    use_freq_spatial_gate = getattr(args, 'use_freq_spatial_gate', False)
    gate_position = getattr(args, 'gate_position', 'after_det_ff')
    gate_mode = getattr(args, 'gate_mode', 'mul')
    use_neck_se = getattr(args, 'use_neck_se', False)
    use_p5_context = getattr(args, 'use_p5_context', False)
    use_freq_attn = getattr(args, 'use_freq_attn', False)
    attn_reduction = int(getattr(args, 'attn_reduction', 16))
    use_det_freq_attn = getattr(args, 'use_det_freq_attn', False)
    det_freq_attn_reduction = int(getattr(args, 'det_freq_attn_reduction', 16))
    det_freq_attn_identity_init = getattr(args, 'det_freq_attn_identity_init', True)
    det_freq_attn_position = getattr(args, 'det_freq_attn_position', 'after_det_ff')
    use_det_dual_freq_attn = getattr(args, 'use_det_dual_freq_attn', False)
    det_dual_freq_beta = float(getattr(args, 'det_dual_freq_beta', 0.1))
    ffparser_num_bands = int(getattr(args, 'ffparser_num_bands', 1))
    ffparser_strategy = getattr(args, 'ffparser_strategy', 'complex')
    dynamic_num_bands = int(getattr(args, 'dynamic_num_bands', 2))
    dynamic_mlp_hidden = int(getattr(args, 'dynamic_mlp_hidden', 64))
    afno_reduction = int(getattr(args, 'afno_reduction', 4))
    spatial_stem_type = getattr(args, 'spatial_stem_type', 'laplacian')
    stem_kernel_size = int(getattr(args, 'stem_kernel_size', 7))
    stem_freeze_diff = getattr(args, 'stem_freeze_diff', False)
    p5_pool_type = getattr(args, 'p5_pool_type', 'avg')
    det_fuse_levels = getattr(args, 'det_fuse_levels', 'p3p4')
    vmamba_cfg = getattr(args, 'vmamba_cfg', None)
    s3mamba_root = getattr(args, 's3mamba_root', None)
    ffparser_mode = getattr(args, 'ffparser_mode', 'det')
    if use_ffparser and ffparser_mode not in ('det', 'fpn', 'both'):
        raise ValueError(
            "ffparser_mode must be 'det', 'fpn', or 'both', got %r" % ffparser_mode
        )
    ffparser_det_hw = None
    ffparser_fpn_hw = None
    if fusion_mode == 'v2':
        use_ffparser = True
        input_side = infer_fsnet_input_side(args, training=training)
        if input_side is not None:
            ffparser_fpn_hw = fpn_scale_spatial_sizes(input_side)
            ffparser_det_hw = det_feat_spatial_size(input_side, use_high_res_fpn)
        if training:
            print(
                'Neck V2: spatial_diff=%s high_res_fpn=%s freq_spatial_gate=%s train_no_patch=%s'
                % (
                    use_spatial_diff,
                    use_high_res_fpn,
                    use_freq_spatial_gate,
                    getattr(args, 'train_no_patch', False),
                )
            )
            if input_side is not None:
                print('  input=%dx%d' % (input_side, input_side))
                p3_hw = ffparser_fpn_hw['p3']
                print(
                    '  cross_scale FFParser (P3): %dx%d'
                    % (p3_hw[0], p3_hw[1])
                )
                print(
                    '  det FFParser: %dx%d'
                    % (ffparser_det_hw[0], ffparser_det_hw[1])
                )
            else:
                print('  FFParser: dynamic HW (no fixed input side)')
    elif fusion_mode == 'crossscale':
        use_ffparser = True
        input_side = infer_fsnet_input_side(args, training=training)
        if input_side is not None:
            ffparser_fpn_hw = fpn_scale_spatial_sizes(input_side)
            if ffparser_mode in ('det', 'both'):
                ffparser_det_hw = det_feat_spatial_size(input_side, use_high_res_fpn)
        if training:
            print(
                'Cross-scale neck: ffparser_mode=%s spatial_diff=%s high_res_fpn=%s'
                % (ffparser_mode, use_spatial_diff, use_high_res_fpn)
            )
            if input_side is not None:
                print('  input=%dx%d' % (input_side, input_side))
                p3_hw = ffparser_fpn_hw['p3']
                print(
                    '  cross_scale FFParser (P3): %dx%d'
                    % (p3_hw[0], p3_hw[1])
                )
                if ffparser_det_hw is not None:
                    print(
                        '  det FFParser: %dx%d'
                        % (ffparser_det_hw[0], ffparser_det_hw[1])
                    )
            else:
                print('  FFParser: dynamic HW (no fixed input side)')
    elif fusion_mode == 'native_spatial':
        use_ffparser = True
        input_side = infer_fsnet_input_side(args, training=training)
        if input_side is not None:
            if ffparser_mode in ('det', 'both'):
                ffparser_det_hw = det_feat_spatial_size(input_side, use_high_res_fpn)
            if ffparser_mode in ('fpn', 'both'):
                ffparser_fpn_hw = fpn_scale_spatial_sizes(input_side)
        if training:
            print(
                'Native-spatial neck: ffparser_mode=%s spatial_diff=%s high_res_fpn=%s'
                % (ffparser_mode, use_spatial_diff, use_high_res_fpn)
            )
            if input_side is not None:
                print('  input=%dx%d' % (input_side, input_side))
                if ffparser_det_hw is not None:
                    print(
                        '  det_feat FFParser: %dx%d'
                        % (ffparser_det_hw[0], ffparser_det_hw[1])
                    )
                if ffparser_fpn_hw is not None:
                    for name in ('p3', 'p4', 'p5'):
                        h, w = ffparser_fpn_hw[name]
                        print('  FPN %s FFParser: %dx%d' % (name, h, w))
    elif use_ffparser:
        input_side = infer_fsnet_input_side(args, training=training)
        if input_side is not None:
            if ffparser_mode in ('det', 'both'):
                ffparser_det_hw = det_feat_spatial_size(input_side, use_high_res_fpn)
            if ffparser_mode in ('fpn', 'both'):
                ffparser_fpn_hw = fpn_scale_spatial_sizes(input_side)
            if training:
                print(
                    'FFParser mode=%s, input=%dx%d, high_res_fpn=%s'
                    % (ffparser_mode, input_side, input_side, use_high_res_fpn)
                )
                if ffparser_det_hw is not None:
                    print(
                        '  det_feat FFParser: %dx%d'
                        % (ffparser_det_hw[0], ffparser_det_hw[0])
                    )
                if ffparser_fpn_hw is not None:
                    for name in ('p3', 'p4', 'p5'):
                        h, w = ffparser_fpn_hw[name]
                        print('  FPN %s FFParser: %dx%d' % (name, h, h))
    if training:
        print(
            'Optional feature modules: wavelet=%s channel_mamba=%s fpn_smooth=%s vssblock=%s(depth=%d) fusion_mode=%s fusion_vote=%s freq_spatial_gate=%s gate_position=%s gate_mode=%s neck_se=%s p5_context=%s freq_attn=%s attn_reduction=%d det_freq_attn=%s det_freq_attn_reduction=%d det_freq_attn_identity_init=%s det_freq_attn_position=%s det_dual_freq_attn=%s det_dual_freq_beta=%.2f'
            % (use_wavelet, use_channel_mamba, use_fpn_smooth, use_vssblock, vssblock_depth, fusion_mode, fusion_vote, use_freq_spatial_gate, gate_position, gate_mode, use_neck_se, use_p5_context, use_freq_attn, attn_reduction, use_det_freq_attn, det_freq_attn_reduction, det_freq_attn_identity_init, det_freq_attn_position, use_det_dual_freq_attn, det_dual_freq_beta)
        )
    # VGG head uses stride-8 anchor layout; Detectron2/Fusion branch uses stride-16 layout.
    use_stride16 = (
        isinstance(backbone_name, str)
        and (backbone_name.startswith('d2_') or backbone_name == 'fusion_competitive')
    )
    if use_high_res_fpn:
        # VGG-FPN P3 is 1/4 (anchor level 2); Detectron2-FPN P3 is 1/8 (level 3).
        anchor_level = 3 if use_stride16 else 2
    else:
        anchor_level = 4 if use_stride16 else 3
    model = FSNet(
        backbone,
        args.row,
        args.line,
        anchor_level=anchor_level,
        use_spatial_diff=use_spatial_diff,
        use_high_res_fpn=use_high_res_fpn,
        use_ffparser=use_ffparser,
        ffparser_mode=ffparser_mode,
        ffparser_det_hw=ffparser_det_hw,
        ffparser_fpn_hw=ffparser_fpn_hw,
        use_wavelet=use_wavelet,
        use_channel_mamba=use_channel_mamba,
        use_fpn_smooth=use_fpn_smooth,
        use_vssblock=use_vssblock,
        vssblock_depth=vssblock_depth,
        vmamba_cfg=vmamba_cfg,
        s3mamba_root=s3mamba_root,
        fusion_mode=fusion_mode,
        fusion_vote=fusion_vote,
        use_amp_phase=use_amp_phase,
        amp_phase_cutoff=amp_phase_cutoff,
        use_freq_spatial_gate=use_freq_spatial_gate,
        gate_position=gate_position,
        gate_mode=gate_mode,
        use_neck_se=use_neck_se,
        use_p5_context=use_p5_context,
        use_freq_attn=use_freq_attn,
        attn_reduction=attn_reduction,
        use_det_freq_attn=use_det_freq_attn,
        det_freq_attn_reduction=det_freq_attn_reduction,
        det_freq_attn_identity_init=det_freq_attn_identity_init,
        det_freq_attn_position=det_freq_attn_position,
        use_det_dual_freq_attn=use_det_dual_freq_attn,
        det_dual_freq_beta=det_dual_freq_beta,
        ffparser_num_bands=ffparser_num_bands,
        ffparser_strategy=ffparser_strategy,
        dynamic_num_bands=dynamic_num_bands,
        dynamic_mlp_hidden=dynamic_mlp_hidden,
        afno_reduction=afno_reduction,
        spatial_stem_type=spatial_stem_type,
        stem_kernel_size=stem_kernel_size,
        stem_freeze_diff=stem_freeze_diff,
        p5_pool_type=p5_pool_type,
        det_fuse_levels=det_fuse_levels,
    )
    if not training:
        return model

    weight_dict = {'loss_ce': 1, 'loss_point': args.point_loss_coef}
    losses = ['labels', 'points']
    if getattr(args, 'geo_loss_coef', 0.0) > 0:
        weight_dict['loss_geo'] = args.geo_loss_coef
        losses.append('geometry')
    if getattr(args, 'count_loss_coef', 0.0) > 0:
        weight_dict['loss_count'] = args.count_loss_coef
        losses.append('count')
    if getattr(args, 'competition_loss_coef', 0.0) > 0:
        weight_dict['loss_competition'] = args.competition_loss_coef
        losses.append('competition')
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(num_classes, \
                                matcher=matcher, weight_dict=weight_dict, \
                                eos_coef=args.eos_coef, losses=losses, geo_k=getattr(args, 'geo_k', 3),
                                density_adaptive_loss=getattr(args, 'density_adaptive_loss', False),
                                density_ref_count=getattr(args, 'density_ref_count', 80.0),
                                density_gamma=getattr(args, 'density_gamma', 0.5),
                                density_weight_min=getattr(args, 'density_weight_min', 0.6),
                                density_weight_max=getattr(args, 'density_weight_max', 1.8),
                                density_adaptive_mode=getattr(args, 'density_adaptive_mode', 'all'),
                                density_high_thresh=getattr(args, 'density_high_thresh', 150.0),
                                uncertainty_weighting=getattr(args, 'uncertainty_weighting', False))

    return model, criterion