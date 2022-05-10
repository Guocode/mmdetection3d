import torch
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss


@weighted_loss
def dir_cosine_loss(pred, sincos_target):
    if sincos_target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == sincos_target.size()
    loss = 1 - F.cosine_similarity(pred, sincos_target)
    return loss


@LOSSES.register_module()
class DirCosineLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DirCosineLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None,avg_factor=None,reduction_override=None,**kwargs):
        """
        Args:
            pred:
            target:

        Returns:
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        sincos_target = torch.stack([torch.sin(target), torch.cos(target)], dim=-1)

        loss = self.loss_weight * dir_cosine_loss(
            pred,
            sincos_target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss