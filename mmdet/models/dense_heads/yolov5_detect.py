# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import warnings
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, bias_init_with_prob,
                      constant_init, is_norm, normal_init)
from mmcv.runner import force_fp32

from mmdet.core import (build_assigner, build_bbox_coder, build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, multiclass_nms)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class YOLOV5Detect(BaseDenseHead, BBoxTestMixin):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): The number of output channels per scale
            before the final 1x1 layer. Default: (1024, 512, 256).
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        featmap_strides (List[int]): The stride of each scale.
            Should be in descending order. Default: (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Default: 0.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        loss_cls (dict): Config of classification loss.
        loss_conf (dict): Config of confidence loss.
        loss_xy (dict): Config of xy coordinate loss.
        loss_wh (dict): Config of wh coordinate loss.
        train_cfg (dict): Training config of YOLOV3 head. Default: None.
        test_cfg (dict): Testing config of YOLOV3 head. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels=(128, 256, 512),
                 out_channels=(128, 256, 512),
                 anchor_generator=dict(type='YOLOAnchorGenerator',
                                       base_sizes=[[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)],
                                                   [(10, 13), (16, 30), (33, 23)]],
                                       strides=[32, 16, 8]),
                 bbox_coder=dict(type='YOLOV5BBoxCoder'),
                 featmap_strides=[32, 16, 8],
                 lcls_weight=0.5, # 对标原yolov5
                 lbox_weight=0.05, # 对标原yolov5
                 lobj_weight=1.0, # 对标原yolov5
                 anchor_thr=4.0, # 对标原yolov5
                 cls_pw=1.0, # 对标原yolov5
                 obj_pw=1.0, # 对标原yolov5
                 one_hot_smoother=0.,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 #  act_cfg=dict(type='SiLU', negative_slope=0.1),
                 act_cfg=dict(type='SiLU'),
                 loss_cls=dict(type='CrossEntropyLoss',
                               use_sigmoid=True, loss_weight=1.0),
                 loss_conf=dict(type='CrossEntropyLoss',
                                use_sigmoid=True, loss_weight=1.0),
                 loss_xy=dict(type='CrossEntropyLoss',
                              use_sigmoid=True, loss_weight=1.0),
                 loss_wh=dict(type='MSELoss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(type='Kaiming',
                               layer='Conv2d',
                               a=0,
                               distribution='normal',
                               mode='fan_out',
                               nonlinearity='relu')):
        super(YOLOV5Detect, self).__init__(init_cfg)
        # Check params
        assert (len(in_channels) == len(out_channels) == len(featmap_strides))
        self.anchors = torch.Tensor([[[3.62500, 2.81250], [4.87500, 6.18750], [11.65625, 10.18750]],
                                     [[1.87500, 3.81250], [3.87500, 2.81250],
                                         [3.68750, 7.43750]],
                                     [[1.25000, 1.62500], [2.00000, 3.75000], [4.12500, 2.87500]]]) # 对标原yolov5
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        self.one_hot_smoother = one_hot_smoother

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.prior_generator = build_prior_generator(anchor_generator)

        self.loss_cls = build_loss(loss_cls)
        self.loss_conf = build_loss(loss_conf)
        self.loss_xy = build_loss(loss_xy)
        self.loss_wh = build_loss(loss_wh)
        self.lcls_weight=lcls_weight
        self.lbox_weight=lbox_weight
        self.lobj_weight=lobj_weight
        self.cls_pw=cls_pw
        self.obj_pw=obj_pw
        self.anchor_thr=anchor_thr
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        assert len(self.prior_generator.num_base_priors) == len(
            featmap_strides)
        self._init_layers()

    @property
    def anchor_generator(self):

        warnings.warn('DeprecationWarning: `anchor_generator` is deprecated, '
                      'please use "prior_generator" instead')
        return self.prior_generator

    @property
    def num_anchors(self):
        """
        Returns:
            int: Number of anchors on each point of feature map.
        """
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, '
                      'please use "num_base_priors" instead')
        return self.num_base_priors

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _init_layers(self):
        self.convs_pred_YOLOV5 = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Conv2d(
                self.out_channels[i], self.num_base_priors * self.num_attrib, 1)
            self.convs_pred_YOLOV5.append(conv_pred)

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             normal_init(m, mean=0, std=0.01)
    #         if is_norm(m):
    #             constant_init(m, 1)

    #     # Use prior in model initialization to improve stability
    #     for conv_pred, stride in zip(self.convs_pred_YOLOV5, self.featmap_strides):
    #         bias = conv_pred.bias.reshape(self.num_base_priors, -1)
    #         # init objectness with prior of 8 objects per feature map
    #         # refer to https://github.com/ultralytics/yolov3
    #         nn.init.constant_(bias.data[:, 4], bias_init_with_prob(8 / (608 / stride)**2))
    #         nn.init.constant_(bias.data[:, 5:], bias_init_with_prob(0.01))

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m,nn.Conv2d):
    #             pass
    #         elif isinstance(m,nn.BatchNorm2d):
    #             m.eps = 1e-3
    #             m.momentum = 0.03

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        assert len(feats) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            x = feats[i]
            pred_map = self.convs_pred_YOLOV5[i](x)
            pred_maps.append(pred_map)
        pred_maps.reverse()
        return tuple(pred_maps),

    @force_fp32(apply_to=('pred_maps', ))
    def get_bboxes(self, pred_maps, img_metas, cfg=None, rescale=False, with_nms=True):
        """Transform network output for a batch into bbox predictions.
        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        result_list = []
        num_levels = len(pred_maps)
        for img_id in range(len(img_metas)):
            pred_maps_list = [pred_maps[i][img_id].detach()
                              for i in range(num_levels)]
            scale_factor = img_metas[img_id]['scale_factor']
            if 'letter_pad' in img_metas[img_id]:
                letter_pad = img_metas[img_id]['letter_pad']
            else:
                letter_pad = None
            if 'pad_duijie' in img_metas[img_id]:
                pad_duijie = img_metas[img_id]['pad_duijie']
            else:
                pad_duijie = None
            proposals = self._get_bboxes_single(
                pred_maps_list, scale_factor, cfg, rescale, letter_pad, pad_duijie,with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self, pred_maps_list, scale_factor, cfg, rescale=False, letter_pad=None, pad_duijie=None,with_nms=True):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(pred_maps_list) == self.num_levels
        multi_lvl_bboxes = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []
        num_levels = len(pred_maps_list)
        featmap_sizes = [pred_maps_list[i].shape[-2:]
                         for i in range(num_levels)]
        multi_lvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, pred_maps_list[0][0].device)
        for i in range(self.num_levels):
            # get some key info for current scale
            pred_map = pred_maps_list[i]
            stride = self.featmap_strides[i]

            # (h, w, num_anchors*num_attrib) -> (h*w*num_anchors, num_attrib)
            pred_map = pred_map.permute(1, 2, 0).reshape(-1, self.num_attrib)

            pred_map[..., :2] = torch.sigmoid(pred_map[..., :2])
            bbox_pred = self.bbox_coder.decode(
                multi_lvl_anchors[i], pred_map[..., :4], stride)
            # conf and cls
            conf_pred = torch.sigmoid(pred_map[..., 4]).view(-1)
            # Cls pred one-hot.
            cls_pred = torch.sigmoid(
                pred_map[..., 5:]).view(-1, self.num_classes)

            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            # conf_inds = conf_pred.ge(conf_thr).nonzero().flatten()
            conf_inds = torch.nonzero(conf_pred.ge(conf_thr)).flatten()
            bbox_pred = bbox_pred[conf_inds, :]
            cls_pred = cls_pred[conf_inds, :]
            conf_pred = conf_pred[conf_inds]

            # Get top-k prediction
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < conf_pred.size(0):
                _, topk_inds = conf_pred.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                cls_pred = cls_pred[topk_inds, :]
                conf_pred = conf_pred[topk_inds]

            # Save the result of current scale
            multi_lvl_bboxes.append(bbox_pred)
            multi_lvl_cls_scores.append(cls_pred)
            multi_lvl_conf_scores.append(conf_pred)

        # Merge the results of different scales together
        multi_lvl_bboxes = torch.cat(multi_lvl_bboxes)
        multi_lvl_cls_scores = torch.cat(multi_lvl_cls_scores)
        multi_lvl_conf_scores = torch.cat(multi_lvl_conf_scores)

        if with_nms and multi_lvl_conf_scores.size(0) == 0:
            return torch.zeros((0, 5)), torch.zeros((0, ))

        if rescale:
            # 先去掉pad，如果有的话
            if letter_pad is not None:
                multi_lvl_bboxes -= multi_lvl_bboxes.new_tensor(
                    [letter_pad[2], letter_pad[0], letter_pad[2], letter_pad[0]])
            # 对接测试
            if pad_duijie is not None:
                multi_lvl_bboxes -= multi_lvl_bboxes.new_tensor(
                    # pad_duijie = []
                    #  左上x            左上y           右下x           右下y
                    [pad_duijie[0], pad_duijie[1], pad_duijie[0], pad_duijie[1]])
            multi_lvl_bboxes /= multi_lvl_bboxes.new_tensor(scale_factor)
            multi_lvl_bboxes = multi_lvl_bboxes.clip(0,640)

        # In mmdet 2.x, the class_id for background is num_classes.
        # i.e., the last column.
        padding = multi_lvl_cls_scores.new_zeros(
            multi_lvl_cls_scores.shape[0], 1)
        multi_lvl_cls_scores = torch.cat(
            [multi_lvl_cls_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(multi_lvl_bboxes,
                                                    multi_lvl_cls_scores,
                                                    cfg.score_thr,
                                                    cfg.nms,
                                                    cfg.max_per_img,
                                                    score_factors=multi_lvl_conf_scores)
            return det_bboxes, det_labels
        else:
            return (multi_lvl_bboxes, multi_lvl_cls_scores, multi_lvl_conf_scores)

    def bbox_iou(self, box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box2 = box2.T

        # Get the coordinates of bounding boxes
        if x1y1x2y2:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  # transform from xywh to xyxy
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union
        if CIoU or DIoU or GIoU:
            # convex (smallest enclosing box) width
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
            ch = torch.max(b1_y2, b2_y2) - \
                torch.min(b1_y1, b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw**2 + ch**2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2)**2) / 4  # center distance squared
                if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi**2) * \
                        torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
                return iou - rho2 / c2  # DIoU
            c_area = cw * ch + eps  # convex area
            # GIoU https://arxiv.org/pdf/1902.09630.pdf
            return iou - (c_area - union) / c_area
        return iou  # IoU

    @force_fp32(apply_to=('pred_maps', ))
    def loss(self, pred_maps, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [pred_maps[i].shape[-2:]
                         for i in range(self.num_levels)]
        # 1.transform: pred_maps --> original yolov5's p of function--build_targets()
        # pred_maps = [
        #     pred_map.view(num_imgs, -1, featmap_sizes[i][0], featmap_sizes[i][1], self.num_attrib).contiguous() for
        #     i, pred_map in
        #     enumerate(pred_maps)
        # ]
        pred_maps = [
            pred_map.view(num_imgs, 3, self.num_attrib, featmap_sizes[i][0],
                          featmap_sizes[i][1]).permute(0, 1, 3, 4, 2).contiguous()
            for i, pred_map in enumerate(pred_maps)
        ]
        # 2.get init targets: [image_idx, gt_label, x, y, w, h]
        #  2.1 gt_bbox: (x1,y1,x2,y2) --> normalized (x,y,w,h)
        gt_bboxes_copy = copy.deepcopy(gt_bboxes)
        bis = img_metas[0]['batch_input_shape']  # (h,w)
        for i in range(len(gt_bboxes)):
            gt_bboxes[i][:, 0] = (
                gt_bboxes_copy[i][:, 0] + gt_bboxes_copy[i][:, 2]) / 2 / bis[1]  # x
            gt_bboxes[i][:, 1] = (
                gt_bboxes_copy[i][:, 1] + gt_bboxes_copy[i][:, 3]) / 2 / bis[0]  # y
            gt_bboxes[i][:, 2] = (
                gt_bboxes_copy[i][:, 2] - gt_bboxes_copy[i][:, 0]) / bis[1]  # w
            gt_bboxes[i][:, 3] = (
                gt_bboxes_copy[i][:, 3] - gt_bboxes_copy[i][:, 1]) / bis[0]  # h
        #  2.2 concat(img_idx,label,bbox) --> targets: [image_idx, gt_label, x, y, w, h]
        targets = []
        for i in range(num_imgs):
            # if gt_labels[i].size()[-1] == 0:
            #     continue
            cls = gt_labels[i][:, None]
            img_idx = torch.full_like(cls, i)
            target = torch.cat((img_idx, cls, gt_bboxes[i]), dim=1)
            targets.append(target)
        targets = torch.cat(targets)
        # get class targets, bbox targets, indices, anchors
        tcls, tbox, indices, anchors = self.build_targets(pred_maps, targets)
        # compute loss
        device = pred_maps[0].device
        # define init loss
        lcls = torch.zeros(1, device=device)  # class loss
        lbox = torch.zeros(1, device=device)  # box loss
        lobj = torch.zeros(1, device=device)
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.cls_pw], device=device))
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.obj_pw], device=device))
        balance = [0.4, 1.0, 4.0]

        for i, pi in enumerate(pred_maps):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(
                pi.shape[:4], dtype=pi.dtype, device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.num_classes),
                                                           1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2)**2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # iou(prediction, target)
                iou = self.bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, 0.0, device=device)  # targets
                    t[range(n), tcls[i]] = 1.0
                    lcls += BCEcls(pcls, t)
            obji = BCEobj(pi[..., 4], tobj)
            lobj += obji * balance[i]  # obj loss
        # bbox loss
        loss_bbox = lbox * self.lbox_weight * num_imgs
        # cls loss
        loss_cls = lcls * self.lcls_weight * num_imgs
        # obj loss
        loss_obj = lobj * self.lobj_weight * num_imgs

        loss_dict = dict(loss_cls=loss_cls,
                         loss_bbox=loss_bbox, loss_obj=loss_obj)

        return loss_dict

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        device = targets.device
        # number of anchors, targets
        na, nt = self.num_anchors, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=device)  # normalized to gridspace gain
        ai = torch.arange(na, device=device).float().view(
            na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # append anchor indices
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
            ],
            device=device).float() * g  # offsets

        for i in range(self.num_levels):
            anchors = self.anchors[i].to(device)
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # scale = self.featmap_strides[i]
            # gain[2:6] = torch.tensor([1 / scale, 1 / scale, 1 / scale, 1 / scale])
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.anchor_thr  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # (image, class), grid xy, grid wh, anchors
            bc, gxy, gwh, a = t.unsafe_chunk(4, dim=1)
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            # image, anchor, grid indices
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def get_targets(self, anchor_list, responsible_flag_list, gt_bboxes_list, gt_labels_list):
        """Compute target maps for anchors in multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_total_anchors, 4).
            responsible_flag_list (list[list[Tensor]]): Multi level responsible
                flags of each image. Each element is a tensor of shape
                (num_total_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.

        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - target_map_list (list[Tensor]): Target map of each level.
                - neg_map_list (list[Tensor]): Negative map of each level.
        """
        pass

    def _get_targets_single(self, anchors, responsible_flags, gt_bboxes, gt_labels):
        """Generate matching bounding box prior and converted GT.

        Args:
            anchors (list[Tensor]): Multi-level anchors of the image.
            responsible_flags (list[Tensor]): Multi-level responsible flags of
                anchors
            gt_bboxes (Tensor): Ground truth bboxes of single image.
            gt_labels (Tensor): Ground truth labels of single image.

        Returns:
            tuple:
                target_map (Tensor): Predication target map of each
                    scale level, shape (num_total_anchors,
                    5+num_classes)
                neg_map (Tensor): Negative map of each scale level,
                    shape (num_total_anchors,)
        """

        pass

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)

    @force_fp32(apply_to=('pred_maps'))
    def onnx_export(self, pred_maps, img_metas, with_nms=True):
        pass
