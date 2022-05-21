import torch
from mmcv.ops import sigmoid_focal_loss
from mmdet.models import weighted_loss
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss
from torch import nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@weighted_loss
def poly1_focal_loss(logits, labels, epsilon=1.0, gamma=2.0):
    p = torch.sigmoid(logits)
    onehot_labels = F.one_hot(labels,logits.size(-1))
    pt = onehot_labels * p + (1 - onehot_labels) * (1 - p)
    FL = py_sigmoid_focal_loss(logits,onehot_labels,reduction='none')
    FLPoly1 = FL + epsilon * torch.pow(1 - pt, gamma + 1)
    return FLPoly1


@LOSSES.register_module()
class PolyFocalLoss(nn.Module):
    """Calculate the focal loss

    Args:
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', gamma=2.0, beta=4.0, alpha=-1):
        super(PolyFocalLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = self.loss_weight * poly1_focal_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss


if __name__ == '__main__':
    ls = PolyFocalLoss()
    a = torch.asarray([[0.1, 0.9,0.]]).cuda()
    b = torch.asarray([0]).long().cuda()
    print(ls(a, b,reduction_override='mean'))

    a = torch.asarray([[9999, -9999.0,-9999.]]).cuda()
    b = torch.asarray([0]).long().cuda()
    print(ls(a, b,reduction_override='mean'))
