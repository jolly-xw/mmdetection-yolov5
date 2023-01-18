import logging
from argparse import ArgumentParser

import torch
import torch.nn as nn

from mmcv.runner import load_checkpoint

from mmdet.models import SingleStageDetector

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def main():
    '''
    another at yuml/models/yolo.py
    '''
    # create model
    v5_backbone = dict(type='YOLOV5Backbone', input_channels=3, return_index=[4, 6, 9], depth_multiple=0.33,
                       width_multiple=0.5)
    v5_neck = None
    v5_head = dict(type='YOLOV5Head', num_classes=80, in_channels=[32, 16, 8], dep_and_wid=[0.33, 0.5])
    model = SingleStageDetector(backbone=v5_backbone, neck=v5_neck, bbox_head=v5_head)
    # get state_dict of model
    model_dict = model.state_dict()

    # # calculate_weights_bias
    # num = 0
    # for key, value in model_dict.items():
    #     if 'weight' in key:
    #         num += 1
    #         print(value.shape)
    #     if 'bias' in key:
    #         num += 1
    #         print(value.shape)
    # print(num)

    # load pretrained checkpoint
    pre_dict = torch.load('model_tools/mm_model/yolov5s_v6_0_mm.pth')
    # update state_dict of model
    model_dict.update(pre_dict['state_dict'])
    model.load_state_dict(model_dict)
    # mm_dict = model.state_dict() # check
    print('loading checkpoint is OK!')

    # input
    x = torch.ones([1, 3, 640, 640])
    y = model.forward_dummy(x)

    print('OK')


if __name__ == '__main__':
    main()
