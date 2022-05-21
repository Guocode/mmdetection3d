import torch
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmcv.ops import boxes_iou_bev, diff_iou_rotated_3d
from mmdet.models.losses.utils import weighted_loss
import numpy as np

from mmdet3d.core.bbox.structures import rotation_3d_in_axis


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


@weighted_loss
def iou_3d_loss(pred, target):
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    iou3d = torch.nan_to_num(diff_iou_rotated_3d(pred[None, ...], target[None, ...]).squeeze(0), 0)
    loss = 1 - iou3d
    return loss


@weighted_loss
def diou_3d_loss(pred, target, eps=1e-6):
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    # pred = torch.cat([pred[:,:6],target[:,6:]],dim=-1)#not opt rot
    iou3d = torch.nan_to_num(diff_iou_rotated_3d(pred[None, ...], target[None, ...]).squeeze(0), 0)
    ctd = torch.sum((pred[:, :3] - target[:, :3]) ** 2, dim=-1)
    cnd = torch.mean(torch.sum((corners_from_bboxes3d(pred) - corners_from_bboxes3d(target)) ** 2, dim=-1), dim=1)
    did = torch.sum(target[:, 3:6] ** 2, dim=-1)
    loss = 1 - iou3d + (ctd + cnd) / (ctd + cnd+ did + eps)#TODO when far diff too small
    return loss


@LOSSES.register_module()
class IOU3DLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(IOU3DLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """
        Args:
            pred: (N, 3+3+1) box (x,y,z,w,h,l,alpha).
            target: (N, 3+3+1) box (x,y,z,w,h,l,alpha).

        Returns:
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = self.loss_weight * iou_3d_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss


@LOSSES.register_module()
class DIOU3DLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, eps=1e-6):
        super(DIOU3DLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """
        Args:
            pred: (N, 3+3+1) box (x,y,z,w,h,l,alpha).
            target: (N, 3+3+1) box (x,y,z,w,h,l,alpha).

        Returns:
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = self.loss_weight * diou_3d_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            eps=self.eps,
            **kwargs
        )
        return loss
