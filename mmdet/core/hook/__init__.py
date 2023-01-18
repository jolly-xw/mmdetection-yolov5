# Copyright (c) OpenMMLab. All rights reserved.
from .checkloss_hook import CheckInvalidLossHook
from .ema import ExpMomentumEMAHook, LinearMomentumEMAHook
from .set_epoch_info_hook import SetEpochInfoHook
from .sync_norm_hook import SyncNormHook
from .sync_random_size_hook import SyncRandomSizeHook
from .yolox_lrupdater_hook import YOLOXLrUpdaterHook
from .yolox_mode_switch_hook import YOLOXModeSwitchHook
from .yolov5_lrupdater_hook import YOLOV5LrUpdaterHook, LambdaLrUpdaterHook
from .yolov5_close_aug_hook import YOLOV5ModeSwitchHook

__all__ = [
    'SyncRandomSizeHook', 'YOLOXModeSwitchHook', 'SyncNormHook',
    'ExpMomentumEMAHook', 'LinearMomentumEMAHook', 'YOLOXLrUpdaterHook',
    'CheckInvalidLossHook', 'SetEpochInfoHook', 'YOLOV5LrUpdaterHook',
    'LambdaLrUpdaterHook', 'YOLOV5ModeSwitchHook'
]
