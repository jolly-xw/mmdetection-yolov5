import copy
import math
import warnings
import torch.nn.functional as F
import torch
import numpy as np
from mmcv.runner import force_fp32
from torch import nn

from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from mmdet.core import (build_assigner, build_bbox_coder,
                        build_sampler, images_to_levels,
                        multi_apply, reduce_mean, build_anchor_generator, multiclass_nms, build_prior_generator)
from mmdet import core
from ..builder import build_loss


class YoloV5InheritHead(BaseDenseHead, BBoxTestMixin):
    def __init__(self,
                 num_classes,
                 in_channels,
                 out_channels=(1024, 512, 256),
                 dep_and_wid=[0.33, 0.5],
                 anchor_generator=dict(
                     type='YOLOAnchorGenerator',
                     base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(10, 13), (16, 30), (33, 23)]],
                     strides=[32, 16, 8]),
                 bbox_coder=dict(type='YOLOBBoxCoder'),
                 featmap_strides=[32, 16, 8],
                 one_hot_smoother=0.,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_obj=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='CIoULoss',
                     eps=1e-16,
                     loss_weight=5.0),
                 train_cfg=None,
                 test_cfg=None):
        super(YoloV5InheritHead, self).__init__()
        assert (len(in_channels) == len(featmap_strides))

        self.anchors = torch.Tensor(
            [[[3.62500, 2.81250],
              [4.87500, 6.18750],
              [11.65625, 10.18750]],

             [[1.87500, 3.81250],
              [3.87500, 2.81250],
              [3.68750, 7.43750]],

             [[1.25000, 1.62500],
              [2.00000, 3.75000],
              [4.12500, 2.87500]]]
        )
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dep_and_wid = dep_and_wid
        self.featmap_strides = featmap_strides
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')  # yolo系列不需随机采样等操作
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.one_hot_smoother = one_hot_smoother

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.anchor_generator = build_prior_generator(anchor_generator)

        self.loss_cls = build_loss(loss_cls)
        self.loss_obj = build_loss(loss_obj)
        self.loss_bbox = build_loss(loss_bbox)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        assert len(
            self.anchor_generator.num_base_anchors) == len(featmap_strides)
        self._init_layers()

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _init_layers(self):
        raise NotImplementedError

    def init_weights(self):
        pass

    def forward(self, feats):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        raise NotImplementedError

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
                result_list中的每一元素都是一个2元组
                第一项是（n，5）张量，其中前 4 列是边界框位置（tl_x、tl_y、br_x、br_y）
                和第5列是介于0和1之间的分数。第二项是（n，） 张量，其中每个项都是对应框

        """
        result_list = []
        num_levels = len(pred_maps)
        for img_id in range(len(img_metas)):
            pred_maps_list = [
                pred_maps[i][img_id].detach() for i in range(num_levels)
            ]
            scale_factor = img_metas[img_id]['scale_factor']
            if 'letter_pad' in img_metas[img_id]:
                letter_pad = img_metas[img_id]['letter_pad']
            else:
                letter_pad = None
            proposals = self._get_bboxes_single(pred_maps_list, scale_factor,
                                                cfg, rescale, letter_pad, with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           pred_maps_list,
                           scale_factor,
                           cfg,
                           rescale=False,
                           letter_pad=None,
                           with_nms=True):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(pred_maps_list) == self.num_levels
        multi_lvl_bboxes = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []
        num_levels = len(pred_maps_list)
        featmap_sizes = [
            pred_maps_list[i].shape[-2:] for i in range(num_levels)
        ]
        multi_lvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, pred_maps_list[0][0].device)
        for i in range(self.num_levels):
            # get some key info for current scale
            pred_map = pred_maps_list[i]
            stride = self.featmap_strides[i]

            # (h, w, num_anchors*num_attrib) -> (h*w*num_anchors, num_attrib)
            pred_map = pred_map.permute(1, 2, 0).reshape(-1, self.num_attrib)

            pred_map[..., :2] = torch.sigmoid(pred_map[..., :2])
            bbox_pred = self.bbox_coder.decode(multi_lvl_anchors[i],
                                               pred_map[..., :4], stride)
            # conf and cls
            conf_pred = torch.sigmoid(pred_map[..., 4]).view(-1)
            cls_pred = torch.sigmoid(pred_map[..., 5:]).view(
                -1, self.num_classes)  # Cls pred one-hot.

            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            conf_inds = conf_pred.ge(conf_thr).nonzero().flatten()
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
            return torch.zeros((0, 5)), torch.zeros((0,))

        if rescale:
            # 先去掉pad，如果有的话
            if letter_pad is not None:
                multi_lvl_bboxes -= multi_lvl_bboxes.new_tensor(
                    [letter_pad[2], letter_pad[0], letter_pad[2], letter_pad[0]])
            multi_lvl_bboxes /= multi_lvl_bboxes.new_tensor(scale_factor)

        # In mmdet 2.x, the class_id for background is num_classes.
        # i.e., the last column.
        padding = multi_lvl_cls_scores.new_zeros(multi_lvl_cls_scores.shape[0],
                                                 1)
        multi_lvl_cls_scores = torch.cat([multi_lvl_cls_scores, padding],
                                         dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                multi_lvl_bboxes,
                multi_lvl_cls_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=multi_lvl_conf_scores)
            return det_bboxes, det_labels
        else:
            return (multi_lvl_bboxes, multi_lvl_cls_scores,
                    multi_lvl_conf_scores)

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
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
                if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
                return iou - rho2 / c2  # DIoU
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
        return iou  # IoU

    @force_fp32(apply_to=('pred_maps',))
    def loss(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
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
        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]
        # 1.transform: pred_maps --> original yolov5's p of function--build_targets()
        # pred_maps = [
        #     pred_map.view(num_imgs, -1, featmap_sizes[i][0], featmap_sizes[i][1], self.num_attrib).contiguous() for
        #     i, pred_map in
        #     enumerate(pred_maps)
        # ]
        pred_maps = [
            pred_map.view(num_imgs, 3, self.num_attrib, featmap_sizes[i][0], featmap_sizes[i][1]).permute(0, 1, 3, 4,
                                                                                                          2).contiguous()
            for
            i, pred_map in
            enumerate(pred_maps)
        ]
        # 2.get init targets: [image_idx, gt_label, x, y, w, h]
        #  2.1 gt_bbox: (x1,y1,x2,y2) --> normalized (x,y,w,h)
        gt_bboxes_copy = copy.deepcopy(gt_bboxes)
        bis = img_metas[0]['batch_input_shape']  # (h,w)
        for i in range(len(gt_bboxes)):
            gt_bboxes[i][:, 0] = (gt_bboxes_copy[i][:, 0] + gt_bboxes_copy[i][:, 2]) / 2 / bis[1]  # x
            gt_bboxes[i][:, 1] = (gt_bboxes_copy[i][:, 1] + gt_bboxes_copy[i][:, 3]) / 2 / bis[0]  # y
            gt_bboxes[i][:, 2] = (gt_bboxes_copy[i][:, 2] - gt_bboxes_copy[i][:, 0]) / bis[1]  # w
            gt_bboxes[i][:, 3] = (gt_bboxes_copy[i][:, 3] - gt_bboxes_copy[i][:, 1]) / bis[0]  # h
        #  2.2 concat(img_idx,label,bbox) --> targets: [image_idx, gt_label, x, y, w, h]
        targets = []
        for i in range(num_imgs):
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
        # 1.get bbox which put in CIoU Loss
        # pbox_in_loss = []
        # tbox_in_loss = []
        # pobj_in_loss = []
        # tobj_in_loss = []
        # pcls_in_loss = []
        # tcls_in_loss = []
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        balance = [0.4, 1.0, 4.0]
        for i, pi in enumerate(pred_maps):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.num_classes),
                                                           1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = self.bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                # pbox_in_loss.append(pbox)

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = iou  # iou ratio
                # tobj_in_loss.append(tobj)
                # pobj_in_loss.append(pi[..., 4])

                # Classification
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, 0.0, device=device)  # targets
                    t[range(n), tcls[i]] = 1.0
                    lcls += BCEcls(pcls, t)
                    # pcls_in_loss.append(pcls)
                    # tcls_in_loss.append(t)
            obji = BCEobj(pi[..., 4], tobj)
            lobj += obji * balance[i]  # obj loss
        # pcls_in_loss = torch.cat(pcls_in_loss)
        # tcls_in_loss = torch.cat(tcls_in_loss)
        # bbox loss
        loss_bbox = lbox.squeeze() * num_imgs * 0.05
        # loss_bbox = self.loss_bbox(pbox_in_loss, tbox_in_loss)
        # cls loss
        # loss_cls = self.loss_cls(pcls_in_loss, tcls_in_loss) * num_imgs
        loss_cls = lcls.squeeze() * num_imgs * 0.5
        # obj loss
        loss_obj = lobj.squeeze() * num_imgs * 1.0
        # if len(pobj_in_loss) == 3:
        #     loss_obj = (self.loss_obj(pobj_in_loss[0], tobj_in_loss[0]) * 0.4 + \
        #                 self.loss_obj(pobj_in_loss[1], tobj_in_loss[1]) * 1.0 + \
        #                 self.loss_obj(pobj_in_loss[2], tobj_in_loss[2]) * 4.0) * num_imgs
        # loss_obj = self.loss_obj(torch.cat(pcls_in_loss), torch.cat(tcls_in_loss))
        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)
        return loss_dict
        # device = pred_maps[0][0].device
        # # generate anchors
        # multi_level_anchors = self.anchor_generator.grid_priors(
        #     featmap_sizes, device)
        # # put anchor to feature map size
        # for i, level_anchor in enumerate(multi_level_anchors):
        #     multi_level_anchors[i] = level_anchor / self.featmap_strides[i]
        # # concat each level anchor
        # flatten_anchors = torch.cat(multi_level_anchors)
        # # repeat anchor for number of images
        # # anchor_list = [flatten_anchors for _ in range(num_imgs)]
        # # e.g. In COCO num_attrib=85
        # pred_maps = [
        #     pred_map.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_attrib) for pred_map in pred_maps
        # ]
        #
        # # e.g. shape = 8*25200*85 for COCO (input shape=640*640)
        # flatten_pred_maps = torch.cat(pred_maps, dim=1)
        # # class --> shape = 8*25200*80
        # flatten_cls_preds = flatten_pred_maps[..., 5:]
        # # bbox of format (x,y,w,h) --> shape = 8*25200*4
        # flatten_bbox_preds = flatten_pred_maps[..., :4]
        # # confidence --> shape = 8*25200
        # flatten_objectness = flatten_pred_maps[..., 4]
        # # get delta between anchors and pred_bboxes
        # # flatten_anchor-->(x1,y1,x2,y2)  flatten_bbox_preds-->(x,y,w,h)
        # flatten_bboxes = self.bbox_coder.delta_bbox(flatten_anchors, flatten_bbox_preds)

        # yolo series need --> yolov5 assigner in it
        # responsible_flag_list = []
        # bbox_targets_list = []
        # for img_id in range(len(img_metas)):
        #     # responsible_flag_list.append(
        #     #     self.anchor_generator.responsible_flags(
        #     #         featmap_sizes, gt_bboxes[img_id], device))
        #     responsible_flag, bbox_targets = self.anchor_generator.responsible_flags(
        #         featmap_sizes, gt_bboxes[img_id], device)
        #     responsible_flag_list.append(responsible_flag)
        #     bbox_targets_list.append(bbox_targets)

        # for i, flag in enumerate(responsible_flag_list):
        #     responsible_flag_list[i] = torch.cat(flag)
        # pos_masks = torch.cat(responsible_flag_list, 0)
        # get target (target and delta --> loss())
        # (pos_masks, cls_targets, obj_targets, bbox_targets, num_fg_imgs) = multi_apply(
        #     self._get_target_v5_single,
        #     flatten_cls_preds.detach(),
        #     flatten_objectness.detach(),
        #     flatten_anchors.unsqueeze(0).repeat(num_imgs, 1, 1),
        #     flatten_bboxes.detach(),
        #     gt_bboxes,
        #     gt_labels
        # )

        # In YOLOx: 'reduce_mean' can improve performance on COCO datasets.
        # num_pos = torch.tensor(
        #     sum(pos_masks),
        #     dtype=torch.float,
        #     device=flatten_cls_preds.device
        # )
        # num_total_samples = max(reduce_mean(num_pos), 1.0)
        #
        # # concat for each image
        # cls_targets = torch.cat(cls_targets, 0)
        # obj_targets = torch.cat(obj_targets, 0)
        # bbox_targets = torch.cat(bbox_targets, 0)
        #
        # # compute loss
        # loss_bbox = self.loss_bbox(
        #     flatten_bboxes.view(-1, 4)[pos_masks],  # deltas
        #     bbox_targets  # targets
        # ) / num_total_samples
        #
        # loss_obj = self.loss_obj(
        #     flatten_objectness.view(-1, -1),
        #     obj_targets
        # ) / num_total_samples
        # losses_cls = self.loss_cls(
        #     flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
        #     cls_targets
        # ) / num_total_samples

        # loss_dict = dict(loss_cls=loss_bbox, loss_bbox=loss_obj, loss_obj=losses_cls)
        # loss_dict = ['pass']

    #
    # return loss_dict

    @torch.no_grad()
    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        device = targets.device
        na, nt = self.num_anchors, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=device)  # normalized to gridspace gain
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

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
                j = torch.max(r, 1 / r).max(2)[0] < 4.0  # compare
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
            bc, gxy, gwh, a = t.unsafe_chunk(4, dim=1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    # @torch.no_grad()
    # def _get_target_v5_single(self,
    #                           cls_preds,
    #                           objectness,
    #                           priors,  # anchor
    #                           decoded_bboxes,  # delta bbox
    #                           responsible_flag,
    #                           gt_bboxes,
    #                           gt_labels):
    #     """
    #     Compute classification, regression, and objectness targets for priors in a single image
    #     """
    #     num_priors = priors.size(0)
    #     num_gts = gt_bboxes.size(0)
    #     gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
    #     # No target
    #     if num_gts == 0:
    #         cls_target = cls_preds.new_zeros((0, self.num_classes))
    #         bbox_target = cls_preds.new_zeros((0, 4))
    #         obj_target = cls_preds.new_zeros((num_priors, 1))
    #         foreground_mask = cls_preds.new_zeros(num_priors).bool()
    #         return (foreground_mask, cls_target, obj_target,  # foreground -> pos_mask
    #                 bbox_target, 0)
    #
    #     assign_result = self.assigner.assign(
    #         responsible_flag, priors, gt_bboxes, gt_labels
    #     )
    #
    #     # unfinished
    #     foreground_mask = ['pass']
    #     cls_target = ['pass']
    #     obj_target = ['pass']
    #     bbox_target = ['pass']
    #     num_pos_per_img = ['pass']
    #     return (foreground_mask, cls_target, obj_target,
    #             bbox_target, num_pos_per_img)

    def get_targets(self, anchor_list, responsible_flag_list, gt_bboxes_list,
                    gt_labels_list, img_metas):
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
        num_imgs = len(anchor_list)

        # anchor number of multi levels 每一层anchor的个数，例如输出是10x10，那么10x10x3=300
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        results = multi_apply(self._get_targets_single, anchor_list,
                              responsible_flag_list, gt_bboxes_list,
                              gt_labels_list, img_metas, num_level_anchors=num_level_anchors)

        all_target_maps, all_neg_maps = results
        assert num_imgs == len(all_target_maps) == len(all_neg_maps)
        # all_target_maps是图片级别list结构，假设有8张图片则len()=8，需要转化为fpn级别的输出，才方便计算Loss,假设有三层输出，则len()=3
        target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
        neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)

        return target_maps_list, neg_maps_list

    @torch.no_grad()
    def _get_targets_single(self, anchors, responsible_flags, gt_bboxes,
                            gt_labels, img_meta, num_level_anchors=None):
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

        anchor_strides = []
        for i in range(len(anchors)):
            anchor_strides.append(
                torch.tensor(self.featmap_strides[i],
                             device=gt_bboxes.device).repeat(len(anchors[i])))
        concat_anchors = torch.cat(anchors)  # 三个输出层的anchor合并
        concat_responsible_flags = torch.cat(responsible_flags)

        anchor_strides = torch.cat(anchor_strides)
        assert len(anchor_strides) == len(concat_anchors) == \
               len(concat_responsible_flags)
        assign_result = self.assigner.assign(concat_anchors,
                                             concat_responsible_flags,
                                             gt_bboxes)

        # 相当于没有，只是为了不报错
        sampling_result = self.sampler.sample(assign_result, concat_anchors,
                                              gt_bboxes)

        # 转化为最终计算Loss所需要的格式
        target_map = concat_anchors.new_zeros(
            concat_anchors.size(0), self.num_attrib)  # 5+class_count

        # 正样本位置anchor bbox；对应的gt bbox；strides
        # target_map前4个是xywh在特征图尺度上面的转化后的label
        target_map[sampling_result.pos_inds, :4] = self.bbox_coder.encode(
            sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes,
            anchor_strides[sampling_result.pos_inds])

        target_map[sampling_result.pos_inds, 4] = 1  # confidence label

        gt_labels_one_hot = F.one_hot(
            gt_labels, num_classes=self.num_classes).float()
        if self.one_hot_smoother != 0:  # label smooth
            gt_labels_one_hot = gt_labels_one_hot * (
                    1 - self.one_hot_smoother
            ) + self.one_hot_smoother / self.num_classes
        target_map[sampling_result.pos_inds, 5:] = gt_labels_one_hot[
            sampling_result.pos_assigned_gt_inds]  # class one hot label

        neg_map = concat_anchors.new_zeros(
            concat_anchors.size(0), dtype=torch.uint8)
        neg_map[sampling_result.neg_inds] = 1

        return target_map, neg_map

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
