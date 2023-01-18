from mmdet.models import CSPDarknetV5, Darknet
import torch

# csp_darknetv5 = CSPDarknetV5(arch='YOLOv5', deepen_factor=0.33, widen_factor=0.5)
input = torch.rand([1, 3, 640, 640])
# res1 = csp_darknetv5.forward(input)
# print('========CSPDarknet V5========')
# # for i in range(len(res1)):
# #     print('第{}层输出:'.format(i + 1), res1[i].shape)

# print(csp_darknetv5)
# from mmdet.models.backbones import YOLOV5Backbone

# v5_backbone = YOLOV5Backbone(depth_multiple=0.33, width_multiple=0.5)
# res2 = v5_backbone.forward(input)
# print('========Oringinal V5========')
# # for i in range(len(res2)):
# #     print('第{}层输出:'.format(i + 1), res2[i].shape)
# print(v5_backbone)

v3_backbone = Darknet(depth=53)
res3 = v3_backbone.forward(input)
print('========Darknet53========')
for i in range(len(res3)):
    print('第{}层输出:'.format(i + 1), res3[i].shape)