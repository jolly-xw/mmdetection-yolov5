from mmdet.core import anchor


dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './model_tools/mm_model/yolov5_6_0/yolov5m_mm.pth'
resume_from = None
_yolo_type = 'm'
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
workflow = [('train', 1)]
model = dict(
    type='YOLOV5',
    pretrained=None,
    backbone=dict(type='CSPDarknetV5', deepen_factor=_depth_multiple, widen_factor=_width_multiple),
    neck=dict(type='YOLOV5Neck', deepen_factor=_depth_multiple, widen_factor=_width_multiple),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0,
        conf_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.3,
            min_pos_iou=0)),
    bbox_head=dict(
        type='YOLOV5Detect',
        num_classes=20,
        out_channels=out_channels,
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOV5BBoxCoder'),
        featmap_strides=[32, 16, 8],
        anchor_thr=2.91,
        cls_pw=0.631,
        obj_pw=0.911,
        lcls_weight=0.243,
        lbox_weight=0.0296,
        lobj_weight=0.301
        ))
data_root = '../datasets/VOC2007train_val/VOCdevkit/'
dataset_type = 'VOCDataset'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True)
img_scale = (640, 640)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize',img_scale=img_scale,keep_ratio=True),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255.0, 255.0, 255.0],
                to_rgb=True),
            dict(type='Pad', pad_to_square=True,pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg'))
        ])
]
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.102, 1.898),
        max_rotate_degree=0.373,
        max_translate_ratio=0.25,
        max_shear_degree=0.602,
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        skip_filter=False),
    # dict(
    #     type='MixUp',
    #     img_scale=img_scale,
    #     ratio_range=(0.8, 1.6),
    #     pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=[0.5,0.0085],direction=['horizontal', 'vertical']),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(
        type='Normalize',
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        to_rgb=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='VOCDataset',
        ann_file='/data/xuew/Xue_Wei/code/MyProject-MMDet/datasets/VOC2007train_val/VOCdevkit/VOC2007/ImageSets/Main/train.txt',
        img_prefix='/data/xuew/Xue_Wei/code/MyProject-MMDet/datasets/VOC2007train_val/VOCdevkit/VOC2007/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False),
    pipeline=train_pipeline)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type='VOCDataset',
        ann_file='/data/xuew/Xue_Wei/code/MyProject-MMDet/datasets/VOC2007train_val/VOCdevkit/VOC2007/ImageSets/Main/val.txt',
        img_prefix='/data/xuew/Xue_Wei/code/MyProject-MMDet/datasets/VOC2007train_val/VOCdevkit/VOC2007/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                                   'img_shape', 'pad_shape', 'scale_factor',
                                   'flip', 'flip_direction', 'img_norm_cfg'))
                ])
        ]),
    test=dict(
        pipeline=test_pipeline
    ))
custom_hooks = [
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
optimizer = dict(
    type='SGD',
    lr=0.0004,
    momentum=0.86,
    weight_decay=0.00035,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='YOLOV5',
    warmup='exp',
    lr0=0.001,
    lrf=0.12,
    by_epoch=True,
    warmup_by_epoch=True,
    warmup_bias_lr=0.004,
    warmup_iters=2,
    warmup_momentum=0.5,
    momentum=0.86)
runner = dict(type='EpochBasedRunner', max_epochs=200)
evaluation = dict(interval=2, metric=['mAP'])
checkpoint_config = dict(interval=200)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
auto_resume = False
gpu_ids = [0]
