'''拿着我的经过ema的yolov5模型去跑inference是会报错的
框可以画出来，但是会显示一串的预训练模型和我的网络不完全匹配'''
import torch

state_dict = torch.load(
    '/home/wx/MyProject-MMDet/MyProject/yuml/finetune_hm_aug_6/latest.pth')
s = torch.load(
    '/home/wx/MyProject-MMDet/MyProject/yuml/finetune_hm_aug_4/latest.pth')
yolox = torch.load('/home/wx/MyProject-MMDet/MyProject/yuml/model_tools/mm_model/yolox_s_8x8_ddc_voc.pth')
pass
