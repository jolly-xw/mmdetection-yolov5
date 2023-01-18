import warnings
import torch.nn.functional as F
import torch
import numpy as np
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from mmdet.core import (build_assigner, build_bbox_coder,
                        build_sampler, images_to_levels,
                        multi_apply, build_anchor_generator, multiclass_nms, build_prior_generator)
from mmdet import core
from ..builder import build_loss


def show_pos_anchor(img_meta, gt_anchors, gt_bboxes, is_show=True):
    # 显示正样本
    assert 'img' in img_meta, print('Collect类中的meta_keys需要新增‘img’，用于可视化调试')
    img = img_meta['img'].data.numpy()
    mean = img_meta['img_norm_cfg']['mean']
    std = img_meta['img_norm_cfg']['std']
    # 默认输入是rgb数据，需要切换为bgr显示
    img = np.transpose(img.copy(), (1, 2, 0))[..., ::-1]
    img = img * std.reshape([1, 1, 3])[..., ::-1] + mean.reshape([1, 1, 3])[..., ::-1]
    img = img.astype(np.uint8)
    img = core.show_bbox(img, gt_anchors.cpu().numpy(), is_show=False)
    return core.show_bbox(img, gt_bboxes.cpu().numpy(), color=(255, 255, 255), is_show=is_show)


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
                 loss_conf=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_xy=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_wh=dict(type='MSELoss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(YoloV5InheritHead, self).__init__()
        assert (len(in_channels) == len(featmap_strides))

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dep_and_wid = dep_and_wid
        self.featmap_strides = featmap_strides
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # self.debug = True
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')  # yolo系列不需随机采样等操作
            self.sampler = build_sampler(sampler_cfg, context=self)
            # self.debug = self.train_cfg.debug

        self.one_hot_smoother = one_hot_smoother

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.anchor_generator = build_prior_generator(anchor_generator)

        self.loss_cls = build_loss(loss_cls)
        self.loss_conf = build_loss(loss_conf)
        self.loss_xy = build_loss(loss_xy)
        self.loss_wh = build_loss(loss_wh)
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
        device = pred_maps[0][0].device

        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]
        multi_level_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        responsible_flag_list = []
        for img_id in range(len(img_metas)):
            # 计算当前图片中的bbox中心在特征图上面的位置,以flag形式表示
            responsible_flag_list.append(
                self.anchor_generator.responsible_flags(
                    featmap_sizes, gt_bboxes[img_id], device))

        target_maps_list, neg_maps_list = self.get_targets(
            anchor_list, responsible_flag_list, gt_bboxes, gt_labels, img_metas)

        losses_cls, losses_conf, losses_xy, losses_wh = multi_apply(
            self.loss_single, pred_maps, target_maps_list, neg_maps_list)

        return dict(
            loss_cls=losses_cls,
            loss_conf=losses_conf,
            loss_xy=losses_xy,
            loss_wh=losses_wh)

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

        # if self.debug:
        #     # 统计下正样本个数
        #     print('----anchor分配正负样本后，正样本anchor可视化，白色bbox是gt----')
        #     gt_inds = assign_result.gt_inds  # 0 1 -1
        #     index = gt_inds > 0
        #     gt_inds = gt_inds[index]
        #     gt_anchors = concat_anchors[index]
        #     print('单张图片中正样本anchor个数', len(gt_inds))
        #     if num_level_anchors is None:  # 不分层显示
        #         show_pos_anchor(img_meta, gt_anchors, gt_bboxes)
        #     else:
        #         imgs = []
        #         count = 0
        #         for num_level in num_level_anchors:
        #             gt_inds = assign_result.gt_inds[count:count + num_level]
        #             anchor = concat_anchors[count:count + num_level]
        #             count += num_level
        #             index = gt_inds > 0
        #             gt_anchor = anchor[index]
        #             img = show_pos_anchor(img_meta, gt_anchor, gt_bboxes, is_show=False)
        #             imgs.append(img)
        #         print('从小特征图到大特征图顺序显示')
        #         cv_core.show_img(imgs)

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
