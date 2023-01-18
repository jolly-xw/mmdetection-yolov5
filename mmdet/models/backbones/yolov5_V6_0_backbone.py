import functools
import math

from torch import nn
from ..builder import BACKBONES

from ..utils.yolov5_V6_0_need import Conv, C3, SPPF


# control width
def _make_divisible(x, divisor, width):
    # control width
    return math.ceil(x * width / divisor) * divisor


def make_divisible(div, wid):
    # refer param
    return functools.partial(_make_divisible, divisor=div, width=wid)


# control depth
def _make_round(n, gd):
    # control depth
    return max(round(n * gd), 1) if n > 1 else n


def make_round(dep):
    # refer param
    return functools.partial(_make_round, gd=dep)


@BACKBONES.register_module()
class YOLOV5Backbone(nn.Module):
    def __init__(self, input_channels=3, return_index=[4, 6, 9], depth_multiple=1.0, width_multiple=1.0):
        # yolov5s --> depth_multiple=0.33, width_multiple=0.5
        super(YOLOV5Backbone, self).__init__()
        self.depth_multiple = depth_multiple  # control depth
        self.width_multiple = width_multiple  # control width
        self.return_index = return_index
        make_div_fun = make_divisible(div=8, wid=self.width_multiple)
        make_round_fun = make_round(dep=self.depth_multiple)

        # create backbone
        model = []

        conv1 = Conv(input_channels, make_div_fun(64), k=6, s=2, p=2)  # 0  args = [3,32,6,2,2]
        model.append(conv1)
        conv2 = Conv(make_div_fun(64), make_div_fun(128), k=3, s=2)  # 1 args = [32,64,3,2]
        model.append(conv2)
        csp1 = C3(make_div_fun(128), make_div_fun(128), n=make_round_fun(3))  # 2 args = [64,64,1]
        model.append(csp1)
        conv3 = Conv(make_div_fun(128), make_div_fun(256), k=3, s=2)  # 3 args = [64,128,3,2]
        model.append(conv3)
        csp2 = C3(make_div_fun(256), make_div_fun(256), n=make_round_fun(6))  # 4 args = [128,128,2] out
        model.append(csp2)
        conv4 = Conv(make_div_fun(256), make_div_fun(512), k=3, s=2)  # 5 args = [128,256,3,2]
        model.append(conv4)
        csp3 = C3(make_div_fun(512), make_div_fun(512), n=make_round_fun(9))  # 6 args = [256,256,3] out
        model.append(csp3)
        conv5 = Conv(make_div_fun(512), make_div_fun(1024), k=3, s=2)  # 7 args = [256,512,3,2]
        model.append(conv5)
        csp4 = C3(make_div_fun(1024), make_div_fun(1024), n=make_round_fun(3))  # 8 args = [512,512,1]
        model.append(csp4)
        sppf = SPPF(make_div_fun(1024), make_div_fun(1024), k=5)  # 9 args = [512,512,5] out
        model.append(sppf)

        self.backbone = nn.Sequential(*model)

    def forward(self, x):
        out = []
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i in self.return_index:
                out.append(x)
        return tuple(out)  # 从大到小
