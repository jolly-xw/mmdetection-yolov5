import logging
import torch
from collections import OrderedDict
from mmdet.models import YOLOV5
from argparse import ArgumentParser

import os

os.chdir('/data/xuew/Xue_Wei/code/MyProject-MMDet/MyProject/yuml/')


def parse_args():
    parser = ArgumentParser(description='官方yolov5模型转换成mmdet可读模型')
    parser.add_argument('--official_weghts', default="model_tools/pt_model/yolov5_6_0/yolov5s.pt", help='官方yolov5模型')
    parser.add_argument('--output_path', default="model_tools/mm_model/yolov5_6_0/yolov5s_mm.pth", help='转换后保存的路径')
    parser.add_argument('--yolov5_size', default="s", help='模型大小,n,s,l,m,x')
    parser.add_argument('--datset_type', default='COCO', help='数据集类型/类别信息,可选VOC,其他的自己改')
    args = parser.parse_args()
    return args


def ori_2_mmdet(args):
    official_weight = args.official_weghts
    save_path = args.output_path
    assert args.yolov5_size in ['n', 's', 'm', 'l', 'x']  # 只有这五种大小
    unneed_key = ['24.anchors', '24.anchor_grid']  # 只有从github官方下载的预训练模型才会有anchor_grid这个key
    if args.yolov5_size == 'n':
        deepen_factor, widen_factor = 0.33, 0.25
        out_channel = (64, 128, 256)
    elif args.yolov5_size == 's':
        deepen_factor, widen_factor = 0.33, 0.5
        out_channel = (128, 256, 512)
    elif args.yolov5_size == 'm':
        deepen_factor, widen_factor = 0.67, 0.75
        out_channel = (192, 384, 768)
    elif args.yolov5_size == 'l':
        deepen_factor, widen_factor = 1.0, 1.0
        out_channel = (256, 512, 1024)
    else:  # x
        deepen_factor, widen_factor = 1.33, 1.25
        out_channel = (320, 640, 1280)
    if args.datset_type == 'COCO':
        class_info = {
            "CLASSES":
            ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush')
        }
    else:
        class_info = {
            "CLASSES":
            ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
             'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
        }
    # 搭建我的MMDetection模型
    backbone = dict(type='CSPDarknetV5', deepen_factor=deepen_factor, widen_factor=widen_factor)
    neck = dict(type='YOLOV5Neck', deepen_factor=deepen_factor, widen_factor=widen_factor)
    head = dict(type='YOLOV5Detect', out_channels=out_channel, num_classes=80)
    model = YOLOV5(backbone=backbone, neck=neck, bbox_head=head)
    # 获取我的模型的state dict
    state_dict = model.state_dict()
    # 去读取我的state dict的所有key
    new_state_dict = OrderedDict()  # 准备一个新的state dict 用于存放我的模型的key，value随便放一个None
    # 遍历我的state dict，把key全部存进新的state dict
    for key, _ in state_dict.items():
        new_state_dict[key] = None

    # 官方模型的pt文件路径
    weights_file = official_weight
    # 加载官方pt文件
    weight = torch.load(weights_file, map_location=torch.device('cpu'))['model']
    ori_model = weight.model
    # 获取官方模型的state dict
    ori_state_dict = ori_model.state_dict()
    # 不去动我的state_dict中的key，去把官方模型的state dict的value赋到我的state_dict中
    # 官方的state dict中是有两个key不需要的：anchors和anchor_grid
    # 去掉不需要的键值对
    # ori_state_dict.pop('24.anchors')
    # ori_state_dict.pop('24.anchor_grid') # 只有官方提供的在COCO上的预训练模型才会有anchor_grid这个key
    for i in range(len(unneed_key)):
        ori_state_dict.pop(unneed_key[i])
    values = []
    for key, value in ori_state_dict.items():
        values.append(value)

    # 前提：确定模型的组件顺序一定没有问题！
    assert len(ori_state_dict) == len(state_dict)  # 判断是否可以一一对应
    for idx, key_value in enumerate(new_state_dict.items()):
        key = key_value[0]
        new_state_dict[key] = values[idx]

    information = class_info  # 这里放类别信息
    data = {"meta": information, "state_dict": new_state_dict}
    logging.info("模型转换成功:original yolov5%s.pt -----> mmdet yolov5%s_mm.pth" % (args.yolov5_size, args.yolov5_size))
    torch.save(data, save_path)


def main():
    args = parse_args()
    ori_2_mmdet(args)


if __name__ == '__main__':
    main()
