import torch
from mmdet.models import YOLOV5

backbone = dict(type='CSPDarknetV5', deepen_factor=0.33, widen_factor=0.5)
neck = dict(type='YOLOV5Neck', deepen_factor=0.33, widen_factor=0.5)
head = dict(type='YOLOV5Detect', num_classes=80)
model = YOLOV5(backbone=backbone, neck=neck, bbox_head=head)
print(model)
# state dict
state_dict = model.state_dict()
x = torch.ones([8, 3, 640, 640])
res = model.forward_dummy(x)

print("===============MMDet—YOLOV5============")
print("给定输入:", x.shape)
for i, r in enumerate(res[0]):
    print(f'第{i+1}层输出为:', r.shape)
