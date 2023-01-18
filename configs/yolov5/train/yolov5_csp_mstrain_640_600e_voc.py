custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(type='YOLOV5',
             pretrained=None,
             backbone=dict(type='CSPDarknetV5', deepen_factor=0.33, widen_factor=0.5),
             neck=dict(type='YOLOV5Neck', deepen_factor=0.33, widen_factor=0.5),
             test_cfg=dict(nms_pre=1000,
                           min_bbox_size=0,
                           score_thr=0,
                           conf_thr=0.001,
                           nms=dict(type='nms', iou_threshold=0.6),
                           max_per_img=300),
             train_cfg=dict(assigner=dict(type='GridAssigner', pos_iou_thr=0.5, neg_iou_thr=0.3, min_pos_iou=0)),
             bbox_head=dict(type='YOLOV5Detect',
                            num_classes=20,
                            anchor_generator=dict(type='YOLOAnchorGenerator',
                                                  base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                                              [(30, 61), (62, 45), (59, 119)],
                                                              [(10, 13), (16, 30), (33, 23)]],
                                                  strides=[32, 16, 8]),
                            bbox_coder=dict(type='YOLOV5BBoxCoder'),
                            featmap_strides=[32, 16, 8]))
data_root = '../datasets/VOC2007train_val/VOCdevkit/'
dataset_type = 'VOCDataset'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True)
img_scale = (640, 640)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(640, 640),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='Normalize', mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect',
                  keys=['img'],
                  meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                             'flip_direction', 'img_norm_cfg'))
         ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type='VOCDataset',
        ann_file=data_root + 'VOC2007/ImageSets/Main/train.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
            dict(type='Expand', mean=[0, 0, 0], to_rgb=True, ratio_range=(1, 2)),
            dict(type='MinIoURandomCrop', min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9), min_crop_size=0.3),
            dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(type='VOCDataset',
             ann_file=data_root + 'VOC2007/ImageSets/Main/val.txt',
             img_prefix=data_root + 'VOC2007/',
             pipeline=[
                 dict(type='LoadImageFromFile'),
                 dict(type='MultiScaleFlipAug',
                      img_scale=(640, 640),
                      flip=False,
                      transforms=[
                          dict(type='Resize', keep_ratio=True),
                          dict(type='Normalize', mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True),
                          dict(type='Pad', size_divisor=32),
                          dict(type='ImageToTensor', keys=['img']),
                          dict(type='Collect',
                               keys=['img'],
                               meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
                                          'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg'))
                      ])
             ]))
optimizer = dict(type='SGD', lr=0.001, momentum=0.937, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', warmup='linear', warmup_iters=2000, warmup_ratio=0.1, step=[218, 450])
runner = dict(type='EpochBasedRunner', max_epochs=600)
evaluation = dict(interval=20, metric=['mAP'])
checkpoint_config = dict(interval=50)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
