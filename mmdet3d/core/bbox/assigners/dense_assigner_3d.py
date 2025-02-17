# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch

# from ..builder import BBOX_ASSIGNERS
# from ..match_costs import build_match_cost
# from ..transforms import bbox_cxcywh_to_xyxy
# from .assign_result import AssignResult
# from .base_assigner import BaseAssigner
from mmcv.ops import diff_iou_rotated_3d
from mmdet.core import BaseAssigner, AssignResult, bbox_cxcywh_to_xyxy
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost

from mmdet3d.core.bbox.structures import rotation_3d_in_axis

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def corners_from_bboxes3d(bboxes3d):
    '''
        bboxes3d: [N,7]
        :return [N,8,3]
    '''
    dims = bboxes3d[:, 3:6].reshape(-1, 1, 3)
    corners_norm = torch.asarray(
        [[-0.5, -0.5, -0.5],
         [-0.5, -0.5, 0.5],
         [-0.5, 0.5, 0.5],
         [-0.5, 0.5, -0.5],
         [0.5, -0.5, -0.5],
         [0.5, -0.5, 0.5],
         [0.5, 0.5, 0.5],
         [0.5, 0.5, -0.5]], dtype=bboxes3d.dtype, device=bboxes3d.device).reshape(1, 8, 3)
    corners = corners_norm * dims
    corners = rotation_3d_in_axis(corners, bboxes3d[:, 6], axis=1)
    corners += bboxes3d[:, :3].view(-1, 1, 3)
    return corners


def diou_3d_cost(pred, target, eps=1e-6):
    # if target.numel() == 0:
    #     return pred.sum() * 0

    from mmdet3d.core import bbox_overlaps_3d
    iou3d = torch.nan_to_num(bbox_overlaps_3d(pred, target), 0)
    ctd = torch.sum((pred[:, :3][:, None, :] - target[:, :3][None, :, :]) ** 2, dim=-1)
    cnd = torch.mean(
        torch.sum((corners_from_bboxes3d(pred)[:, None, :] - corners_from_bboxes3d(target)[None, :, :]) ** 2, dim=-1),
        dim=-1)
    did = torch.sum(target[:, 3:6] ** 2, dim=-1)[None, :]
    cost = -iou3d + (ctd + cnd) / (did + eps)
    return cost


def bboxes_iou3d(boxes1, boxes2):
    """Calculate 3D overlaps of two boxes.
    Args:
        boxes1 N*7
        boxes2 M*7
    Returns:
        torch.Tensor: M*N
    """
    from mmdet3d.core import bbox_overlaps_3d
    overlaps_3d = torch.nan_to_num(bbox_overlaps_3d(boxes1, boxes2), 0)
    volume1 = boxes1[:, 3:6].prod(-1, keepdims=True)
    volume2 = boxes2[:, 3:6].prod(-1, keepdims=True)
    iou3d = overlaps_3d / torch.clamp(volume1 + volume2 - overlaps_3d, min=1e-8)
    return iou3d


def dis_3d_cost(pred, target, eps=1e-6):
    # if target.numel() == 0:
    #     return pred.sum() * 0
    from mmdet3d.core import bbox_overlaps_3d
    # iou3d = bboxes_iou3d(pred, target)
    iou3d = torch.nan_to_num(bbox_overlaps_3d(pred, target, 'iou'), 0)
    ctd = torch.sum((pred[:, :3][:, None, :] - target[:, :3][None, :, :]) ** 2, dim=-1)
    # cnd = torch.mean(torch.sum((corners_from_bboxes3d(pred)[:,None,:] - corners_from_bboxes3d(target)[None,:,:]) ** 2, dim=-1), dim=-1)
    did = torch.sum(target[:, 3:6] ** 2, dim=-1)[None, :]
    diou = iou3d - ctd / (ctd + did + eps)  # (-1,1 better)
    cost = -diou  # (-1 better,1)
    return cost


@BBOX_ASSIGNERS.register_module()
class DenseAssigner3D(BaseAssigner):
    """
    Cycle HungarianAssigner3D
    Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 iou_cost=dis_3d_cost, cycassign_decay=0.7):
        self.cls_cost = build_match_cost(cls_cost)
        self.iou_cost = iou_cost
        self.cycassign_decay = cycassign_decay

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w, _ = img_meta['img_shape']
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression iou cost, defaultly giou is used in official DETR.
        # bbox_pred = torch.cat([bbox_pred[:,:6],torch.atan2(bbox_pred[:,6:7],bbox_pred[:,7:8])],dim=-1)
        iou_cost = self.iou_cost(bbox_pred, gt_bboxes)
        # weighted sum of above three costs
        l1_cost  = torch.cdist(bbox_pred[:,:3], gt_bboxes[:,:3], p=1)
        cost = cls_cost + l1_cost
        cost_ = cost.detach().cpu().clone()
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        assigned_num = 0
        assigned_cyc = 0
        max_overlaps = torch.zeros(bbox_pred.size(0), dtype=torch.float32, device=bbox_pred.device)
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        # cycled assign until all preds are assigned
        while assigned_num < bbox_pred.size(0):
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost_)
            matched_row_inds = torch.from_numpy(matched_row_inds).to(
                bbox_pred.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(
                bbox_pred.device)
            valid_assign = min(bbox_pred.size(0)-assigned_num,gt_bboxes.size(0))
            matched_row_inds = matched_row_inds.flip(0)[:valid_assign]
            matched_col_inds = matched_col_inds.flip(0)[:valid_assign]
            assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
            assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
            max_overlaps[matched_row_inds] = (1 - (
                    (iou_cost[matched_row_inds, matched_col_inds]+1)/2)).clamp(0.,1.) * (self.cycassign_decay ** assigned_cyc)
            # make the assigned unassignable
            cost_[matched_row_inds] = 9999
            assigned_num += gt_bboxes.size(0)
            assigned_cyc += 1

        # 4. assign backgrounds and foregrounds
        # assign foregrounds based on matching results
        # assigned_gt_inds = pred_assign + 1
        # assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        max_overlaps = max_overlaps.to(bbox_pred.device)
        return AssignResult(
            bbox_pred.size(0), assigned_gt_inds, max_overlaps, labels=assigned_labels)
