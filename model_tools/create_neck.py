from mmdet.models import YOLOV5Neck, YOLOV5Head2, YOLOV3Neck
import torch

feats = []
x = torch.ones([1, 128, 80, 80])
y = torch.ones([1, 256, 40, 40])
z = torch.ones([1, 512, 20, 20])
feats.append(x)
feats.append(y)
feats.append(z)
feats = tuple(feats)
yolov5_neck = YOLOV5Neck(deepen_factor=0.33, widen_factor=0.5)
print('====================MMDet--YOLOV5Neck===================')
for j, x in enumerate(feats):
    print(f"第{j+1}层输入shape为:", x.shape)

res = yolov5_neck.forward(feats)
print(res)
for i, r in enumerate(res):
    print(f"第{i+1}层输出shape为:", r.shape)
print(yolov5_neck)

# original_yolov5 = YOLOV5Head2(num_classes=80, in_channels=[32, 16, 8], dep_and_wid=[0.33, 0.5])
# print('===================Original--YOLOV5Neck=================')
# res = original_yolov5.forward(feats)
# print(res)

# feats = []
# x = torch.ones([1, 256, 80, 80])
# y = torch.ones([1, 512, 40, 40])
# z = torch.ones([1, 1024, 20, 20])
# feats.append(x)
# feats.append(y)
# feats.append(z)
# feats = tuple(feats)
# YOLOV3neck = YOLOV3Neck(3, in_channels=[1024, 512, 256], out_channels=[512, 256, 128])
# res = YOLOV3neck.forward(feats)
# print(res)