import functools
import math

from torch import nn
from ..builder import HEADS

from ..utils.yolov5_V6_0_need import Conv, C3, Concat
from .yolov5_head_inherit import YoloV5InheritHead


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


@HEADS.register_module()
class YOLOV5Head(YoloV5InheritHead):

    def _init_layers(self):
        make_div_fun = make_divisible(8, self.dep_and_wid[1])
        make_round_fun = make_round(self.dep_and_wid[0])
        model = []

        conv1 = Conv(make_div_fun(1024), make_div_fun(512), k=1, s=1)  # 10 args = [512,256,1,1]
        model.append(conv1)
        up_sample1 = nn.Upsample(None, scale_factor=2, mode='nearest')  # 11 args = [None,2,'nearest']
        model.append(up_sample1)
        concat1 = Concat(1)  # 12 args = [1]
        model.append(concat1)
        csp1 = C3(make_div_fun(512) + make_div_fun(512), make_div_fun(512), n=make_round_fun(3), shortcut=False)
        model.append(csp1)  # 13 args = [512,256,1,False]

        conv2 = Conv(make_div_fun(512), make_div_fun(256), k=1, s=1)  # 14 args = [256,128,1,1]
        model.append(conv2)
        up_sample2 = nn.Upsample(None, scale_factor=2, mode='nearest')  # 15 args = [None,2,'nearest']
        model.append(up_sample2)
        concat2 = Concat(1)  # 16 args = [1]
        model.append(concat2)
        csp2 = C3(make_div_fun(256) + make_div_fun(256), make_div_fun(256), n=make_round_fun(3), shortcut=False)
        model.append(csp2)  # 17 args = [256,128,1,False]

        conv3 = Conv(make_div_fun(256), make_div_fun(256), k=3, s=2)  # 18 args = [128,128,3,2]
        model.append(conv3)
        concat3 = Concat(1)  # 19 args = [1]
        model.append(concat3)
        csp3 = C3(make_div_fun(256) + make_div_fun(256), make_div_fun(512), n=make_round_fun(3), shortcut=False)
        model.append(csp3)  # 20 args = [256,256,1,False]

        conv4 = Conv(make_div_fun(512), make_div_fun(512), k=3, s=2)  # 21 args = [256,256,3,2]
        model.append(conv4)
        concat4 = Concat(1)  # 22 args = [1]
        model.append(concat4)
        csp4 = C3(make_div_fun(512) + make_div_fun(512), make_div_fun(1024), n=make_round_fun(3), shortcut=False)
        model.append(csp4)
        self.det = nn.Sequential(*model)
        self.head = nn.ModuleList([
            nn.Conv2d(make_div_fun(256), 255, 1),
            nn.Conv2d(make_div_fun(512), 255, 1),
            nn.Conv2d(make_div_fun(1024), 255, 1)
        ])

    def forward(self, feats):
        large_feat, inter_feat, small_feat = feats

        small_feat = self.det[0](small_feat)  # conv
        x = self.det[1](small_feat)  # UpSample
        x = self.det[2]([x, inter_feat])  # concat
        x = self.det[3](x)  # csp
        inter_feat = self.det[4](x)  # conv

        x = self.det[5](inter_feat)  # UpSample
        x = self.det[6]([x, large_feat])  # concat
        x = self.det[7](x)  # csp
        # print(x.shape) torch.Size([1, 128, 80, 80])
        out0 = self.head[0](x)  # 第一个输出层

        x = self.det[8](x)  # conv
        x = self.det[9]([x, inter_feat])
        x = self.det[10](x)  # csp
        # print(x.shape) torch.Size([1, 256, 40, 40])
        out1 = self.head[1](x)  # 第二个输出层

        x = self.det[11](x)  # conv
        x = self.det[12]([x, small_feat])
        x = self.det[13](x)  # csp
        # print(x.shape) torch.Size([1, 512, 20, 20])
        out2 = self.head[2](x)  # 第三个输出层

        return tuple([out2, out1, out0]),  # 从小到大特征图返回
