import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss

INF = 1e8
EPS = 1e-12
PI = np.pi


@HEADS.register_module()
class DenseDepthHead(nn.Module):
    def __init__(self,
                 in_channel,
                 feat_channel,
                 densedepth_bin,
                 loss_densedepth=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(DenseDepthHead, self).__init__()
        self.norm = nn.BatchNorm2d
        self.densedepth_head = self._build_head(in_channel, feat_channel, densedepth_bin)
        self.loss_densedepth = build_loss(loss_densedepth)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
    def _build_head(self, in_channel, feat_channel, out_channel):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            self._get_norm_layer(feat_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def _get_norm_layer(self, feat_channel):
        return self.norm(feat_channel, momentum=0.03, eps=0.001)

    def init_weights(self):
        bias_init = bias_init_with_prob(0.1)
        self.densedepth_head[-1].bias.data.fill_(bias_init)  # -2.19
        for head in [self.densedepth_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        densedepth_pred = self.densedepth_head(feat)
        return densedepth_pred,
    @force_fp32(apply_to=('center_heatmap_preds', 'depth_preds'))
    def loss(self,
             densedepth_preds,
             densedepth_gts,
             img_metas,
             attr_labels=None,
             proposal_cfg=None):
        loss_densedepth = dict()
        for i, densedepth_pred in enumerate(densedepth_preds):
            densedepth_gt_p = F.adaptive_max_pool2d(densedepth_gts, densedepth_pred.shape[-2:])
            gt_valid = (densedepth_gt_p != 0)
            loss_densedepth['loss_densedepth_lvl_' + str(i)] = self.loss_densedepth(densedepth_pred[gt_valid],
                                                                                    densedepth_gt_p[gt_valid])

        return loss_densedepth

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels, gt_bboxes_3d,
                      gt_labels_3d, centers2d, depths,
                      kpts2d, kpts2d_valid,
                      attr_labels, gt_bboxes_ignore,
                      densedepth=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)

        loss_inputs = outs + (densedepth, img_metas,)
        losses = self.loss(*loss_inputs, )

        if proposal_cfg is None:
            return losses
        else:
            raise NotImplementedError
