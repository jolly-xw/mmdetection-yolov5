# Copyright (c) OpenMMLab. All rights reserved.
import math
from turtle import forward

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from ..utils import CSPLayer

from mmcv.cnn import ACTIVATION_LAYERS

# @ACTIVATION_LAYERS.register_module()
# class SiLU(nn.Module):

#     def __init__(self, inplace=False):
#         super(SiLU, self).__init__()
#         self.act = nn.SiLU(inplace)

#     def forward(self, x):
#         return self.act(x)
ACTIVATION_LAYERS.register_module(module=nn.SiLU, name='SiLU')


class SPPFBottleneck(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels,
                                mid_channels,
                                1,
                                stride=1,
                                padding=0,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        conv2_in = mid_channels * 4
        conv2_out = out_channels
        self.conv2 = ConvModule(conv2_in, conv2_out, 1, stride=1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        pool_pad = kernel_size // 2
        self.pooling = nn.MaxPool2d(kernel_size, stride=1, padding=pool_pad)

    def forward(self, x):
        x = self.conv1(x)
        # with warnings.catch_warning():
        # warning.simplefilter('ignore')
        y1 = self.pooling(x)
        y2 = self.pooling(y1)
        y3 = torch.cat([x, y1, y2, self.pooling(y2)], dim=1)
        out = self.conv2(y3)
        return out


@BACKBONES.register_module()
class CSPDarknetV5(BaseModule):
    """CSP-Darknet backbone used in YOLOv5 and YOLOX.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5, P6}.
            Default: P5.
        deepen_factor (float): Depth multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        widen_factor (float): Width multiplier, multiply number of
            blocks in CSP layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False.
        arch_ovewrite(list): Overwrite default arch settings. Default: None.
        spp_kernal_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        >>> from mmdet.models import CSPDarknet
        >>> import torch
        >>> self = CSPDarknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_sppf
    arch_settings = {
        'YOLOv5': [[64, 128, 3, True, False], [128, 256, 6, True, False], [256, 512, 9, True, False],
                   [512, 1024, 3, True, True]],
    }

    def __init__(self,
                 arch='YOLOv5',
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 sppf_kernal_size=5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU'),
                 norm_eval=False,
                 init_cfg=dict(type='Kaiming',
                               layer='Conv2d',
                               a=0,
                               distribution='normal',
                               mode='fan_out',
                               nonlinearity='relu')):
        super().__init__(init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        self.stem_YOLOV5 = ConvModule(  # 替代原本的Focus
            3,  # 替代focus卷积层的in_channel
            int(64 * widen_factor),  # 替代focus卷积层的out_channel
            kernel_size=6,
            stride=2,
            padding=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.layers = ['stem_YOLOV5']

        for i, (in_channels, out_channels, num_blocks, add_identity, use_sppf) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = conv(in_channels,
                              out_channels,
                              3,
                              stride=2,
                              padding=1,
                              conv_cfg=conv_cfg,
                              norm_cfg=norm_cfg,
                              act_cfg=act_cfg)
            stage.append(conv_layer)
            csp_layer = CSPLayer(out_channels,
                                 out_channels,
                                 num_blocks=num_blocks,
                                 add_identity=add_identity,
                                 use_depthwise=use_depthwise,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=act_cfg)
            stage.append(csp_layer)
            if use_sppf:
                sppf = SPPFBottleneck(out_channels,
                                      out_channels,
                                      kernel_size=sppf_kernal_size,
                                      conv_cfg=conv_cfg,
                                      norm_cfg=norm_cfg,
                                      act_cfg=act_cfg)
                stage.append(sppf)
            self.add_module(f'stage{i + 1}_YOLOV5', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}_YOLOV5')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(CSPDarknetV5, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
