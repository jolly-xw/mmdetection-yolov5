_yolo_type = 'x'
if _yolo_type == 'n':
    _depth_multiple = 0.33
    _width_multiple = 0.25
    out_channels=(64, 128, 256)
elif _yolo_type == 's':
    _depth_multiple = 0.33
    _width_multiple = 0.5
    out_channels=(128, 256, 512)
elif _yolo_type == 'm':
    _depth_multiple = 0.67
    _width_multiple = 0.75
    out_channels=(192, 384, 768)
elif _yolo_type == 'l':
    _depth_multiple = 1.0
    _width_multiple = 1.0
    out_channels=(256, 512, 1024)
elif _yolo_type == 'x':
    _depth_multiple = 1.33
    _width_multiple = 1.25
    out_channels=(320, 640, 1280)
else:
    raise NotImplementedError

model = dict(
    type='YOLOV5',
    pretrained=None,
    backbone=dict(type='CSPDarknetV5', deepen_factor=_depth_multiple, widen_factor=_width_multiple),
    neck=dict(type='YOLOV5Neck', deepen_factor=_depth_multiple, widen_factor=_width_multiple),
    test_cfg=dict(nms_pre=1000,
                  min_bbox_size=0,
                  score_thr=0,
                  conf_thr=0.001,
                  nms=dict(type='nms', iou_threshold=0.6),
                  max_per_img=300),  # 1000
    bbox_head=dict(type='YOLOV5Detect',
                   num_classes=80,
                   out_channels=out_channels,
                   anchor_generator=dict(type='YOLOAnchorGenerator',
                                         base_sizes=[[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45),
                                                                                           (59, 119)],
                                                     [(10, 13), (16, 30), (33, 23)]],
                                         strides=[32, 16, 8]),
                   bbox_coder=dict(type='YOLOV5BBoxCoder'),
                   featmap_strides=[32, 16, 8],
                   loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0, reduction='sum'),
                   loss_conf=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0, reduction='sum'),
                   loss_xy=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=2.0, reduction='sum'),
                   loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')))

img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
img_scale = (640, 640)
# dataset settings
# dataset_type = 'VOCDataset'
# data_root = 'data/VOCdevkit/'
data_root = '../datasets/coco/'
dataset_type = 'CocoDataset'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            dict(type='LetterBox', new_shape=(640, 640)),
            # dict(type='RandomFlip'),
            # 'RandomFlip' is unable, because param(flip=False) and there is no flip_direction in 'MultiScaleFlipAug'
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect',
                 keys=['img'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'letter_pad'))
        ])
]

data = dict(samples_per_gpu=4,
            workers_per_gpu=2,
            persistent_workers=True,
            test=dict(type=dataset_type,
                      ann_file=data_root + 'annotations/instances_val2017.json',
                      img_prefix=data_root + 'images/val2017/',
                      pipeline=test_pipeline))
