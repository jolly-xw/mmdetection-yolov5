# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import torch
import torch.nn.functional as F  # F.interpolate == nn.Upsample
import math
from mmcv.cnn import ConvModule  # ConvModule with SiLU act == CBS
from mmcv.runner import BaseModule

from ..builder import NECKS
# CSPLayer that add_identity is False == C3 of YOLOv5 neck
from ..utils import CSPLayer


@NECKS.register_module()
class YOLOV5Neck(BaseModule):
    """The neck of YOLOV5.

    It can be treated as a simplified version of FPN. It
    will take the result from Darknet backbone and do some upsampling and
    concatenation. It will finally output the detection result.

    Note:
        The input feats should be from top to bottom.
            i.e., from high-lvl to low-lvl
        But YOLOV3Neck will process them in reversed order.
            i.e., from bottom (high-lvl) to top (low-lvl)

    Args:
        num_scales (int): The number of scales / stages.
        in_channels (List[int]): The number of input channels per scale.
        out_channels (List[int]): The number of output channels  per scale.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Dictionary to construct and config norm
            layer. Default: dict(type='BN', requires_grad=True)
        act_cfg (dict, optional): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity
    arch_settings = {
        'YOLOv5': [[1024, 512, 3, False], [512, 256, 3, False], [256, 256, 3, False], [512, 512, 3, False]],
    }

    def __init__(
            self,
            arch='YOLOv5',
            deepen_factor=1.0,
            widen_factor=1.0,
            num_scales=3,  # 猜测是backbone的len(out)
            use_depthwise=False,
            conv_cfg=None,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU'),
            init_cfg=dict(
                type='Kaiming',
                layer='Conv2d',
                a=0,
                distribution='normal',
                mode='fan_out',
                nonlinearity='relu')):
        super(YOLOV5Neck, self).__init__(init_cfg)
        self.num_scales = num_scales
        # default number of YOLOv5 backbone's out
        assert (self.num_scales == 3)
        arch_setting = self.arch_settings[arch]
        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        for i, (in_channels, out_channels, num_blocks, add_identity) in enumerate(arch_setting):
            # in_channels, out_channels, num_blocks, add_identity
            in_channels = int(in_channels * widen_factor)  # s->0.5
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)  # s->0.33
            if i == 0 or i == 1:
                kernel_size, stride, padding, in_c, out_c = 1, 1, 0, in_channels, out_channels
            else:
                kernel_size, stride, padding, in_c, out_c = 3, 2, 1, 2 * \
                    in_channels, 2 * out_channels
            self.add_module(f'neck_conv{i+1}_YOLOV5', ConvModule(in_channels, out_channels, kernel_size, stride, padding,
                                                                 **cfg))
            csp_layer = CSPLayer(in_c,
                                 out_c,
                                 num_blocks=num_blocks,
                                 add_identity=add_identity,
                                 use_depthwise=use_depthwise,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=act_cfg)
            self.add_module(f'neck_c3_{i+1}_YOLOV5', csp_layer)

    def forward(self, feats):
        assert len(feats) == self.num_scales
        outs = []
        cat = [feats[0]]
        conv = getattr(self, f'neck_conv{1}_YOLOV5')
        out = conv(feats[-1])  # torch.Size([1,256,20,20])
        cat_3 = out
        out = F.interpolate(out, scale_factor=2)  # torch.Size([1,256,40,40])
        out = torch.cat((out, feats[1]), 1)  # torch.Size([1,512,40,40])
        csp = getattr(self, f'neck_c3_{1}_YOLOV5')
        out = csp(out)  # torch.Size([1,256,40,40])
        conv = getattr(self, f'neck_conv{2}_YOLOV5')
        out = conv(out)  # torch.Size([1,128,40,40])
        cat.append(out)
        cat.append(cat_3)
        out = F.interpolate(out, scale_factor=2)  # torch.Size([1,128,80,80])

        for i in range(self.num_scales):
            if i > 0:
                conv = getattr(self, f'neck_conv{i+2}_YOLOV5')
                out = conv(out)
            out = torch.cat((out, cat[i]), 1)  # torch.Size([1,256,80,80])
            csp = getattr(self, f'neck_c3_{i+2}_YOLOV5')
            out = csp(out)  # torch.Size([1,128,80,80])
            outs.append(out)

        return tuple(outs)
