import torch

state_dict = torch.load('/home/wx/git/yuml/test_path/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')
print(state_dict)