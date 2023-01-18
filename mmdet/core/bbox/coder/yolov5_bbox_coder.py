import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view(-1, 2).float()


@BBOX_CODERS.register_module()
class YOLOV5BBoxCoder(BaseBBoxCoder):
    """YOLO BBox coder.

    Following `YOLO <https://arxiv.org/abs/1506.02640>`_, this coder divide
    image into grids, and encode bbox (x1, y1, x2, y2) into (cx, cy, dw, dh).
    cx, cy in [0., 1.], denotes relative center position w.r.t the center of
    bboxes. dw, dh are the same as :obj:`DeltaXYWHBBoxCoder`.

    Args:
        eps (float): Min value of cx, cy when encoding.
    """

    def __init__(self, eps=1e-6):
        super(BaseBBoxCoder, self).__init__()
        self.eps = eps

    def encode(self, bboxes, gt_bboxes, stride):
        """
        YOLOv5 and YOLOx of YOLO series don't have encode
        """
        raise NotImplementedError("YOLOv5 doesn't have encoder!")

    def delta_bbox(self, bboxes, pred_bboxes):
        """Get delta_bboxes from anchors and pred_bboxes.

        Args:
            bboxes: anchors.(x1,y1,x2,y2)
            pred_bboxes: output bboxes of YOLOv5.(x,y,w,h)
        """
        # anchors' width and height
        w, h = bboxes[..., 2] - bboxes[..., 0], bboxes[..., 3] - bboxes[..., 1]

        # center of x & y of pred_bbox
        x_center_pred = (pred_bboxes[..., 0].sigmoid() - 0.5) * 2
        y_center_pred = (pred_bboxes[..., 1].sigmoid() - 0.5) * 2
        # w & h of pred_bbox
        w_pred = (pred_bboxes[..., 2].sigmoid() * 2) ** 2 * w
        h_pred = (pred_bboxes[..., 3].sigmoid() * 2) ** 2 * h

        delta_bboxes = torch.stack(
            (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
             x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
            dim=-1
        )

        return delta_bboxes

    def decode(self, bboxes, pred_bboxes, stride):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes, e.g. anchors.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            stride (torch.Tensor | int): Strides of bboxes.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        assert pred_bboxes.size(-1) == bboxes.size(-1) == 4

        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]

        # Get outputs x, y
        # 由于mmdetection的anchor已经偏移了0.5，故*2的操作要放在外面
        x_center_pred = (pred_bboxes[..., 0] - 0.5) * 2 * stride + x_center
        y_center_pred = (pred_bboxes[..., 1] - 0.5) * 2 * stride + y_center
        # yolov5中正常情况应该是
        # x_center_pred = (pred_bboxes[..., 0] * 2. - 0.5 + grid[:, 0]) * stride  # xy
        # y_center_pred = (pred_bboxes[..., 1] * 2. - 0.5 + grid[:, 1]) * stride  # xy

        # wh也需要sigmoid，然后乘以4来还原
        w_pred = (pred_bboxes[..., 2].sigmoid() * 2) ** 2 * w
        h_pred = (pred_bboxes[..., 3].sigmoid() * 2) ** 2 * h

        decoded_bboxes = torch.stack(
            (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
             x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
            dim=-1)

        return decoded_bboxes
