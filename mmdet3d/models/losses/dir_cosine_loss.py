import torch
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class DirCosineLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(DirCosineLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        """

        Args:
            pred:
            target:

        Returns:

        """
        sincos_target = torch.stack([torch.sin(target),torch.cos(target)],dim=-1)
        loss = 1-F.cosine_similarity(pred,sincos_target)
        return loss.mean() * self.loss_weight

