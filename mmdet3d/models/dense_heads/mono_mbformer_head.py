import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import (bbox2distance, build_assigner, build_sampler, distance2bbox,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, )
from mmdet.models.builder import HEADS, build_loss

from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
import math

from ..losses.iou3d_loss import diou_3d_loss
from ...core import CameraInstance3DBoxes

INF = 1e8
EPS = 1e-12
PI = math.pi

class Mobile2Former(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Mobile2Former, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        x =  x.reshape(b, c, h*w).transpose(1,2).unsqueeze(1)
        q = self.to_q(z).view(b, self.heads, m, c)
        dots = q @ x.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        out = attn @ x
        out = rearrange(out, 'b h m c -> b m (h c)')
        return z + self.to_out(out)


# inputs: x(b c h w) z(b m d)
# output: x(b c h w)
class Former2Mobile(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Former2Mobile, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, channel),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        q =  x.reshape(b, c, h*w).transpose(1,2).unsqueeze(1)
        k = self.to_k(z).view(b, self.heads, m, c)
        v = self.to_v(z).view(b, self.heads, m, c)
        dots = q @ k.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        out = attn @ v
        out = rearrange(out, 'b h l c -> b l (h c)')
        out = self.to_out(out)
        out = out.view(b, c, h, w)
        return x + out

class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        shape = x.size()
        x = F.softmax(x.reshape(*shape[:-1], self.reg_max + 1), dim=-1)
        x = F.linear(x, self.project.type_as(x)).reshape(*shape[:-1])
        return x


@HEADS.register_module()
class MonoMBFormerHead(AnchorFreeHead):
    """Generalized Focal Loss V2: Learning Reliable Localization Quality
    Estimation for Dense Object Detection.
    GFocal head structure is similar with GFL head, however GFocal uses
    the statistics of learned distribution to guide the
    localization quality estimation (LQE)
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='GN', num_groups=32, requires_grad=True).
        loss_qfl (dict): Config of Quality Focal Loss (QFL).
        reg_max (int): Max value of integral set :math: `{0, ..., reg_max}`
            in QFL setting. Default: 16.
        reg_topk (int): top-k statistics of distribution to guide LQE
        reg_channels (int): hidden layer unit to generate LQE

    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 loss_gc3d=dict(type='SmoothL1Loss', beta=0.05),
                 loss_offset2c3d=dict(type='SmoothL1Loss', beta=0.05),
                 loss_depth=dict(type='SmoothL1Loss', beta=0.05),
                 loss_dim=dict(type='SmoothL1Loss', beta=0.05),
                 loss_dir=dict(type='DirCosineLoss'),
                 loss_iou3d=dict(type='DIOU3DLoss'),
                 reg_max=16,
                 depth_reg_max=32,
                 depth_max=80,
                 reg_topk=4,
                 reg_channels=64,
                 add_mean=True,
                 aux_reg=False,
                 bbox3d_code_size=7,
                 depth_exprg=9,
                 **kwargs):

        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.total_dim = reg_topk
        self.aux_reg = aux_reg
        self.bbox3d_code_size = bbox3d_code_size
        self.depth_reg_max = depth_reg_max
        self.depth_max = depth_max
        self.depth_exprg = depth_exprg

        print('total dim = ', self.total_dim * 4)

        super(MonoMBFormerHead, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.integral = Integral(self.reg_max)
        self.depth_intergral = Integral(self.depth_reg_max)
        self.loss_dfl = build_loss(loss_dfl)

        self.loss_gc3d = build_loss(loss_gc3d)
        self.loss_offset2c3d = build_loss(loss_offset2c3d)
        self.loss_depth = build_loss(loss_depth)
        self.loss_dim = build_loss(loss_dim)
        self.loss_dir = build_loss(loss_dir)
        self.loss_iou3d = build_loss(loss_iou3d)
        self._init_canon_box_sizes()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg3d_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg3d_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        # assert self.num_anchors == 1, 'anchor free version'

        # self.gfl_cls = nn.Conv2d(
        #     self.feat_channels, self.num_classes, 3, padding=1)
        self.gfl_cls = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.num_classes, 3, padding=1),
            nn.Sigmoid()
        )
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        # 3D
        self.dir_reg = nn.Conv2d(
            self.feat_channels, 2, 3, padding=1)
        self.dim_reg = nn.Conv2d(
            self.feat_channels, 3, 3, padding=1)
        self.depth_reg = nn.Conv2d(
            self.feat_channels, self.depth_reg_max + 1, 3, padding=1)
        self.offset2c3d_reg = nn.Conv2d(
            self.feat_channels, 2, 3, padding=1)
        # aux reg
        if self.aux_reg:
            self.offset2kpts_reg = nn.Conv2d(
                self.feat_channels, 8 * 2, 3, padding=1)
            self.c3dkptshm_reg = nn.Conv2d(
                self.feat_channels, 8 + 1, 3, padding=1)

        # self.scales2d = nn.ModuleList(
        #     [Scale(1.0) for _ in self.strides])
        # self.scales3d_depth = nn.ModuleList(
        #     [Scale(1.0) for _ in self.strides])
        # self.bias3d_depth = nn.ModuleList(
        #     [Bias(-4.0) for _ in self.strides])
        # self.scales3d_offset2c3d = nn.ModuleList(
        #     [Scale(1.0) for _ in self.strides])
        conf_vector = [nn.Conv2d(4 * self.total_dim, self.reg_channels, 1)]
        conf_vector += [self.relu]
        conf_vector += [nn.Conv2d(self.reg_channels, 1, 1), nn.Sigmoid()]

        self.reg_conf = nn.Sequential(*conf_vector)

        self.temperature_scores3d = 1e-1

    def _init_canon_box_sizes(self):
        self.register_buffer('canon_box_sizes', torch.asarray(
            [[1.61876949, 3.89154523, 1.52969237],  # Car
             [0.62806586, 0.82038497, 1.76784787],  # Pedestrian
             [0.56898187, 1.77149234, 1.7237099],  # Cyclist
             [1.9134491, 5.15499603, 2.18998422],  # Van
             [2.61168401, 9.22692319, 3.36492722],  # Truck
             [0.5390196, 1.08098042, 1.28392158],  # Person_sitting
             [2.36044838, 15.56991038, 3.5289238],  # Tram
             [1.24489164, 2.51495357, 1.61402478],  # Misc
             ])[:, [1, 2, 0]])

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg3d_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_conf:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)
        normal_init(self.gfl_reg, std=0.01)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_3d=None,
                      gt_kpts_2d=None,
                      gt_kpts_valid_mask=None,
                      attr_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        assert gt_labels is not None
        assert attr_labels is None
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_bboxes_3d,
                              gt_kpts_2d, gt_kpts_valid_mask,
                              img_metas)
        losses = self.loss(*loss_inputs)

        if proposal_cfg is None:
            return losses
        else:
            raise NotImplementedError

    def forward(self, feats):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification and quality (IoU)
                    joint scores for all scale levels, each is a 4D-tensor,
                    the channel number is num_classes.
                bbox_preds (list[Tensor]): Box distribution logits for all
                    scale levels, each is a 4D-tensor, the channel number is
                    4*(n+1), n is max value of integral set.
        """
        out_cls_score = []
        out_bbox_dist_pred = []
        out_bbox_pred = []
        out_depth_dist_pred = []
        out_depth_pred = []
        out_offset2c3d_pred = []
        out_dim_pred = []
        out_dir_pred = []

        for lvlid, feat in enumerate(feats):
            cls_feat = feat
            reg_feat = feat
            reg3d_feat = feat
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            for reg3d_conv in self.reg3d_convs:
                reg3d_feat = reg3d_conv(reg3d_feat)
            # bbox_dist_pred = self.scales2d[lvlid](self.gfl_reg(reg_feat))
            bbox_dist_pred = self.gfl_reg(reg_feat)
            N, C, H, W = bbox_dist_pred.size()
            bbox_pred = self.integral(bbox_dist_pred.permute(0, 2, 3, 1).reshape(N, H, W, 4, self.reg_max + 1)).permute(
                0, 3, 1, 2)
            prob = F.softmax(bbox_dist_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
            prob_topk, _ = prob.topk(self.reg_topk, dim=2)

            if self.add_mean:
                stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)],
                                 dim=2)
            else:
                stat = prob_topk

            quality_score = self.reg_conf(stat.reshape(N, -1, H, W))
            cls_score = self.gfl_cls(cls_feat) * quality_score
            # cls_score = self.gfl_cls(cls_feat) * quality_score
            # 3D
            # offset2c3d_pred = self.scales3d_offset2c3d[lvlid](self.offset2c3d_reg(reg3d_feat))
            offset2c3d_pred = self.offset2c3d_reg(reg3d_feat)

            depth_dist_pred = self.depth_reg(reg3d_feat)
            depth_pred = self.depth_intergral(
                depth_dist_pred.permute(0, 2, 3, 1).reshape(N, H, W, 1, self.depth_reg_max + 1)).permute(0, 3, 1, 2)
            depth_pred = depth_pred / self.depth_reg_max * self.depth_max  # 0-depth_reg_max

            dim_pred = torch.tanh(self.dim_reg(reg3d_feat))
            dir_pred = self.dir_reg(reg3d_feat)
            # depth_pred[:, 0, :, :] = self.bias3d_depth[lvlid](self.scales3d_depth[lvlid](
            #     1. / (depth_pred[:, 0, :, :].sigmoid() + EPS) - 1))

            # aux
            if self.aux_reg:
                ofset2kpts_pred = self.offset2kpts_reg(reg3d_feat)
                c3dkptshm_pred = self.c3dkptshm_reg(reg3d_feat)

            out_cls_score.append(cls_score.flatten(start_dim=2))
            out_bbox_dist_pred.append(bbox_dist_pred.flatten(start_dim=2))
            out_bbox_pred.append(bbox_pred.flatten(start_dim=2))
            out_offset2c3d_pred.append(offset2c3d_pred.flatten(start_dim=2))
            out_depth_dist_pred.append(depth_dist_pred.flatten(start_dim=2))
            out_depth_pred.append(depth_pred.flatten(start_dim=2))
            out_dim_pred.append(dim_pred.flatten(start_dim=2))
            out_dir_pred.append(dir_pred.flatten(start_dim=2))

        out_cls_scores = torch.cat(out_cls_score, dim=2).permute(0, 2, 1)
        out_bbox_dist_preds = torch.cat(out_bbox_dist_pred, dim=2).permute(0, 2, 1)
        out_bbox_preds = torch.cat(out_bbox_pred, dim=2).permute(0, 2, 1)
        out_offset2c3d_preds = torch.cat(out_offset2c3d_pred, dim=2).permute(0, 2, 1)
        out_depth_dist_preds = torch.cat(out_depth_dist_pred, dim=2).permute(0, 2, 1)
        out_depth_preds = torch.cat(out_depth_pred, dim=2).permute(0, 2, 1)
        out_dim_preds = torch.cat(out_dim_pred, dim=2).permute(0, 2, 1)
        out_dir_preds = torch.cat(out_dir_pred, dim=2).permute(0, 2, 1)

        return out_cls_scores, out_bbox_dist_preds, out_bbox_preds, out_offset2c3d_preds, \
               out_depth_dist_preds, out_depth_preds, out_dim_preds, out_dir_preds

    @force_fp32(apply_to=('out_cls_scores', 'out_bbox_preds', 'out_dir_preds', 'out_dim_preds', 'out_depth_preds',
                          'out_offset2c3d_preds'))
    def loss(self,
             out_cls_scores, out_bbox_dist_preds, out_bbox_preds, out_offset2c3d_preds, \
             out_depth_dist_preds, out_depth_preds, out_dim_preds, out_dir_preds,
             gt_bboxes,
             gt_labels,
             gt_bboxes3d,
             gt_kpts2d,
             gt_kpts2d_valid,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.
        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, num_points).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), num_points), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        device = out_cls_scores.device
        input_height, input_width = img_metas[0]["img_shape"][:2]
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width / stride))
            for stride in self.strides
        ]
        batch_size = out_cls_scores.shape[0]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        multi_lvl_ltcenter_priors = self.get_multi_lvl_ltcenter_priors(batch_size, featmap_sizes, self.strides,
                                                                       dtype=torch.float32,
                                                                       device=out_cls_scores.device)
        # label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        out_bbox_s_preds = out_bbox_preds * multi_lvl_ltcenter_priors[..., 2, None]
        decoded_bboxes = distance2bbox(multi_lvl_ltcenter_priors[..., :2], out_bbox_s_preds)
        decoded_pc3d = multi_lvl_ltcenter_priors[..., :2] + out_offset2c3d_preds * multi_lvl_ltcenter_priors[
            ..., 2, None]  # proj center
        print('offset_mean:',(out_offset2c3d_preds * multi_lvl_ltcenter_priors[ ..., 2, None]).abs().mean())
        cam2imgs = torch.stack([img_meta['cam2img'] for img_meta in img_metas], dim=0).to(device)
        batch_target_res = self.get_targets(out_cls_scores,
                                            multi_lvl_ltcenter_priors,
                                            decoded_bboxes,
                                            gt_bboxes,
                                            gt_labels, gt_bboxes3d, gt_kpts2d, gt_kpts2d_valid)
        loss, loss_states = self.get_loss_from_target(
            out_cls_scores,
            out_bbox_dist_preds,
            out_depth_dist_preds,
            out_depth_preds,
            out_dim_preds,
            out_dir_preds,
            decoded_bboxes, decoded_pc3d,
            batch_target_res,
            cam2imgs
        )
        return loss_states

    def get_bboxes(self,
                   out_cls_scores, out_bbox_dist_preds, out_bbox_preds, out_offset2c3d_preds, \
                   out_depth_dist_preds, out_depth_preds, out_dim_preds, out_dir_preds,
                   img_metas, thresh=0.35):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            img_metas (list): list of batch image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        device = out_cls_scores.device
        batch_size = out_cls_scores.shape[0]
        input_height, input_width = img_metas[0]["img_shape"][:2]
        input_shape = (input_height, input_width)

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width / stride))
            for stride in self.strides
        ]
        # get grid cells of one image
        multi_lvl_ltcenter_priors = self.get_multi_lvl_ltcenter_priors(batch_size, featmap_sizes, self.strides,
                                                                       dtype=torch.float32,
                                                                       device=device)
        out_bbox_s_preds = out_bbox_preds * multi_lvl_ltcenter_priors[..., 2, None]
        decoded_bboxes = distance2bbox(multi_lvl_ltcenter_priors[..., :2], out_bbox_s_preds, max_shape=input_shape)
        # decoded_bboxes = out_bbox_preds
        decoded_c3d = multi_lvl_ltcenter_priors[..., :2] + out_offset2c3d_preds * multi_lvl_ltcenter_priors[
            ..., 2, None]
        # decoded_c3d = out_offset2c3d_preds
        out_dir_preds = torch.atan2(out_dir_preds[..., 0:1], out_dir_preds[..., 1:])
        out_bbox3d_preds = torch.cat([decoded_c3d, out_depth_preds[..., 0:1], out_dim_preds, out_dir_preds], dim=-1)

        result_list = []
        for i in range(batch_size):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox, bbox3d = out_cls_scores[i], decoded_bboxes[i], out_bbox3d_preds[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            bbox, cls, inds = multiclass_nms(
                bbox,
                score,
                score_thr=thresh,
                nms_cfg=dict(type="nms", iou_threshold=0.6),
                max_num=100,
                return_inds=True
            )
            bbox3d = bbox3d[:, None, :].expand(bbox3d.size(0), self.num_classes, self.bbox3d_code_size).reshape(-1,
                                                                                                                self.bbox3d_code_size)[
                inds]
            cam2imgs = img_metas[i]['cam2img'][None,].to(device)
            bbox3d = self.decode_bbox3d(bbox3d, cls.reshape(-1), cam2imgs.repeat([bbox3d.size(0), 1, 1]))
            bbox3d = CameraInstance3DBoxes(bbox3d, box_dim=self.bbox3d_code_size, origin=(0.5, 0.5, 0.5))
            result_list.append([bbox, bbox3d,cls])
        return result_list

    def get_targets(self, cls_preds, multi_lvl_ltcenter_priors, decoded_bboxes, gt_bboxes, gt_labels, gt_bboxes3d,
                    gt_kpts2d, gt_kpts2d_valid):
        batch_assign_res = multi_apply(
            self.target_assign_single_img,
            cls_preds.detach(),
            multi_lvl_ltcenter_priors,
            decoded_bboxes.detach(),
            gt_bboxes,
            gt_labels, gt_bboxes3d, gt_kpts2d, gt_kpts2d_valid
        )
        return batch_assign_res

    @torch.no_grad()
    def target_assign_single_img(
            self, cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels, gt_bboxes3d, gt_kpts2d,
            gt_kpts2d_valid,
    ):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            center_priors (Tensor): All priors of one image, a 2D-Tensor with
                shape [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = center_priors.size(0)
        device = center_priors.device
        gt_bboxes = torch.asarray(gt_bboxes).to(device)
        gt_labels = torch.asarray(gt_labels).to(device)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        if isinstance(gt_bboxes3d, CameraInstance3DBoxes):  # target bbox3d with gravity center and alpha
            # gt_bboxes3d = torch.cat([gt_bboxes3d.gravity_center, gt_bboxes3d.dims, gt_bboxes3d.local_yaw[..., None]],
            #                         dim=-1).to(gt_bboxes.device)
            gt_bboxes3d = torch.cat([gt_bboxes3d.gravity_center, gt_bboxes3d.dims, gt_bboxes3d.yaw[..., None]],
                                    dim=-1).to(gt_bboxes.device)
        single_img_labels = center_priors.new_full(
            (num_priors,), self.num_classes, dtype=torch.long
        )
        single_img_label_scores = center_priors.new_zeros(single_img_labels.shape, dtype=torch.float)
        # single_img_label_scores3d = center_priors.new_zeros(single_img_labels.shape, dtype=torch.float)
        single_img_bbox_targets = torch.empty((0, 4))
        single_img_dist_targets = torch.empty((0, 4))
        single_img_bbox3d_targets = torch.empty((0, 7))
        single_img_depth_dist_targets = torch.empty((0, 1))
        single_img_kpts2d_targets = torch.empty((0, 9, 2))
        single_img_kpts2d_valid_targets = torch.empty((0, 9))
        single_img_pos_inds = torch.empty((0))
        # No target
        if num_gts == 0:
            return single_img_labels, single_img_label_scores, single_img_bbox_targets, single_img_dist_targets, \
                   single_img_bbox3d_targets, single_img_depth_dist_targets, single_img_kpts2d_targets, single_img_kpts2d_valid_targets, single_img_pos_inds

        assign_result = self.assigner.assign(
            cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels
        )
        pos_inds, neg_inds, pos_gt_bboxes, pos_gt_bboxes3d, pos_gt_kpts2d, pos_gt_kpts2d_valid, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes, gt_bboxes3d, gt_kpts2d, gt_kpts2d_valid
        )
        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]
        if len(pos_inds) > 0:
            single_img_bbox_targets = pos_gt_bboxes
            single_img_dist_targets = (bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes) / center_priors[
                pos_inds, None, 2]).clamp(min=0, max=self.reg_max - 0.1)
            single_img_labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            single_img_label_scores[pos_inds] = pos_ious
            # single_img_label_scores3d[pos_inds] = pos_ious#TODO 3d iou
            single_img_bbox3d_targets = pos_gt_bboxes3d
            # lvl_inds = [self.strides.index(stride) for stride in center_priors[pos_inds, 2]]
            # single_img_depth_dist_targets = [
            #     (torch.log(pos_gt_bboxes3d[pos_ind, 2:3].clamp(0.1, 300)) - self.bias3d_depth[
            #         lvl].bias.detach()) / self.depth_exprg * self.reg_max / self.scales3d_depth[lvl].scale.detach()
            #     for pos_ind, lvl in enumerate(lvl_inds)]
            single_img_depth_dist_targets = pos_gt_bboxes3d[:, 2:3] / self.depth_max * self.depth_reg_max
            # print(self.bias3d_depth[1].bias, self.scales3d_depth[1].scale)
            single_img_depth_dist_targets = single_img_depth_dist_targets.clamp(min=0,max=self.depth_reg_max - 0.1)
            single_img_kpts2d_targets = pos_gt_kpts2d
            single_img_kpts2d_valid_targets = pos_gt_kpts2d_valid
            single_img_pos_inds = pos_inds
        return single_img_labels, single_img_label_scores, single_img_bbox_targets, single_img_dist_targets, \
               single_img_bbox3d_targets, single_img_depth_dist_targets, single_img_kpts2d_targets, single_img_kpts2d_valid_targets, single_img_pos_inds

    def decode_bbox3d(self, bbox3d, cls, cam2img):
        '''
        2d+depth to 3d
        tanh dim to dim
        alpha to roty
        :param bbox3d: [N,7]
        :param cls: [N,]
        :param cam2img: [N,3,3]
        :return:
        '''
        #  2d+depth -> 3d
        bbox3d_xyz = self.pts2Dto3D(bbox3d[:, :3], cam2img)
        bbox3d_dims = torch.exp(bbox3d[:, 3:6]) * self.canon_box_sizes[cls]
        #  alpha -> roty
        # bbox3d_roty = (bbox3d[:, 6:] + torch.atan2(bbox3d[:, 0:1] - cam2img[:, 0, 2:3], cam2img[:, 0, 0:1]) + PI) % (
        #         PI * 2) - PI
        bbox3d_roty = bbox3d[:, 6:]
        return torch.cat([bbox3d_xyz,bbox3d_dims,bbox3d_roty],dim=-1)

    def encode_bbox3d(self, bbox3d):
        '''
        roty to alpha
        3d to 2d+depth
        :param bbox3d: [N,7]
        :param cam2img: [N,3,3]
        :return:
        '''
        return

    @staticmethod
    def pts2Dto3D(points, cam2img):
        """
        Args:
            points (torch.Tensor): points in 2D images, [N, 3], \
                3 corresponds with x, y in the image and depth.
            cam2img (torch.Tensor): camera instrinsic, [N,3, 3]

        Returns:
            torch.Tensor: points in 3D space. [N, 3], \
                3 corresponds with x, y, z in 3D space.
        """
        # assert view.shape[0] <= 4
        # assert view.shape[1] <= 4
        # assert points.shape[1] == 3
        # cam2img = cam2img
        points2D = points[:, :2]
        depths = points[:, 2:]  # .view(-1, 1)
        unnorm_points2D = torch.cat([points2D * depths, depths], dim=1)

        cam2img_mono = torch.nn.functional.pad(cam2img, (0, 1, 0, 1))
        cam2img_mono[:, 3, 3] = 1.
        # viewpad[:view.shape[0], :view.shape[1]] = points2D.new_tensor(view)
        inv_cam2img_mono = torch.inverse(cam2img_mono).permute(0, 2, 1).to(points.device)
        # inv_cam2img_mono = torch.linalg.inv(cam2img_mono).permute(1, 0)
        # Do operation in homogenous coordinates.
        nbr_points = unnorm_points2D.shape[0]
        homo_points2D = torch.cat(
            [unnorm_points2D,
             points2D.new_ones((nbr_points, 1))], dim=1).view(-1, 1, 4)  #
        points3D = torch.bmm(homo_points2D, inv_cam2img_mono).view(-1, 4)[:, :3]

        return points3D

    def get_single_lvl_ltcenter_priors(
            self, batch_size, featmap_size, stride, dtype, device
    ):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        y, x = torch.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        proiors = torch.stack([x, y, strides, strides], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)

    def get_multi_lvl_ltcenter_priors(self, batch_size, featmap_sizes, strides, dtype, device):
        '''

        Args:
            batch_size:
            featmap_sizes:
            strides:
            dtype:
            device:

        Returns:
            [left,top,stride,stride]
        '''
        multi_lvl_ltcenter_priors = [
            self.get_single_lvl_ltcenter_priors(
                batch_size,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(strides)
        ]
        multi_lvl_ltcenter_priors = torch.cat(multi_lvl_ltcenter_priors, dim=1)
        return multi_lvl_ltcenter_priors

    def sample(self, assign_result, gt_bboxes, gt_bboxes3d, gt_kpts2d, gt_kpts2d_valid):
        """Sample positive and negative bboxes."""
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
                .squeeze(-1)
                .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
                .squeeze(-1)
                .unique()
        )
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
            pos_gt_bboxes3d = torch.empty_like(gt_bboxes3d).view(-1, 7)
            pos_gt_kpts2d = torch.empty_like(gt_kpts2d).view(-1, 18)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
            pos_gt_bboxes3d = gt_bboxes3d[pos_assigned_gt_inds, :]
            pos_gt_kpts2d = gt_kpts2d[pos_assigned_gt_inds, :]
            pos_gt_kpts2d_valid = gt_kpts2d_valid[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_gt_bboxes3d, pos_gt_kpts2d, pos_gt_kpts2d_valid, pos_assigned_gt_inds

    def get_loss_from_target(self,
                             out_cls_scores,
                             out_bbox_dist_preds,
                             out_depth_dist_preds,
                             out_depth_preds,
                             out_dim_preds,
                             out_dir_preds,
                             decoded_bboxes, decoded_pc3d,
                             batch_target_res,
                             cam2imgs):
        device = out_cls_scores.device
        num_priors = out_cls_scores.size(1)
        batch_labels, batch_label_scores, batch_bbox_targets, batch_dist_targets, \
        batch_bbox3d_targets, batch_depth_dist_targets, batch_kpts2d_targets, batch_kpts2d_valid_targets, batch_pos_inds = batch_target_res
        batch_size = len(batch_labels)
        batch_pos_binds = torch.cat(
            [pos_inds + num_priors * batch_id for batch_id, pos_inds in enumerate(batch_pos_inds)])
        batch_pos_bids = torch.cat(
            [torch.full_like(pos_inds, batch_id) for batch_id, pos_inds in enumerate(batch_pos_inds)])
        num_batch_pos = batch_pos_binds.size(0)

        batch_labels = torch.cat(batch_labels, dim=0)
        batch_label_scores = torch.cat(batch_label_scores, dim=0)
        batch_bbox_targets = torch.cat(batch_bbox_targets, dim=0)
        batch_dist_targets = torch.cat(batch_dist_targets, dim=0)
        batch_bbox3d_targets = torch.cat(batch_bbox3d_targets, dim=0)
        batch_depth_dist_targets = torch.cat(batch_depth_dist_targets, dim=0)
        batch_kpts2d_targets = torch.cat(batch_kpts2d_targets, dim=0)
        batch_kpts2d_valid_targets = torch.cat(batch_kpts2d_valid_targets, dim=0)

        out_cls_scores = out_cls_scores.reshape(-1, self.num_classes)
        out_bbox_dist_preds = out_bbox_dist_preds.reshape(-1, 4 * (self.reg_max + 1))
        out_depth_dist_preds = out_depth_dist_preds.reshape(-1, self.depth_reg_max + 1)
        out_depth_preds = out_depth_preds.reshape(-1, 1)
        out_dim_preds = out_dim_preds.reshape(-1, 3)
        out_dir_preds = out_dir_preds.reshape(-1, 2)

        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        decoded_pc3d = decoded_pc3d.reshape(-1, 2)

        if num_batch_pos > 0:
            decoded_gc3d = self.pts2Dto3D(
                torch.cat([decoded_pc3d[batch_pos_binds], out_depth_preds[batch_pos_binds]], dim=-1),
                cam2imgs[batch_pos_bids])  # gravity center
            # decoded_gc3d_c3d = self.pts2Dto3D(
            #     torch.cat([decoded_pc3d[batch_pos_binds], batch_bbox3d_targets[:, 2:3]], dim=-1),
            #     cam2imgs[batch_pos_bids])  # gravity center
            # decoded_gc3d_depth = torch.cat([batch_bbox3d_targets[:, 0:2], out_depth_preds[batch_pos_binds]], dim=-1)
            out_dim_pos = torch.exp(out_dim_preds[batch_pos_binds]) * self.canon_box_sizes[
                batch_labels[batch_pos_binds]]
            out_dir_pos = out_dir_preds[batch_pos_binds]
            out_dir_rad_pos = torch.atan2(out_dir_pos[..., 0:1], out_dir_pos[..., 1:])
            decoded_bboxes3d = torch.cat([decoded_gc3d, out_dim_pos, out_dir_rad_pos], dim=-1)  # not opt alpha
            # batch_iou3d = torch.nan_to_num(diff_iou_rotated_3d(decoded_bboxes3d[None, ...], batch_bbox3d_targets[None, ...]).squeeze(0), 0)

            # decoded_bboxes3d_wodir = torch.cat([decoded_gc3d, out_dim_pos, batch_bbox3d_targets[:, 6:]],
            #                                    dim=-1)  # not opt alpha
            # decoded_bboxes3d_c3d = torch.cat(
            #     [decoded_gc3d_c3d, batch_bbox3d_targets[:, 3:6], batch_bbox3d_targets[:, 6:]], dim=-1)
            # decoded_bboxes3d_depth = torch.cat(
            #     [decoded_gc3d_depth, batch_bbox3d_targets[:, 3:6], batch_bbox3d_targets[:, 6:]], dim=-1)
            # decoded_bboxes3d_dim = torch.cat([batch_bbox3d_targets[:, :3], out_dim_pos, batch_bbox3d_targets[:, 6:]],
            #                                  dim=-1)
            # decoded_bboxes3d_dir = torch.cat(
            #     [batch_bbox3d_targets[:, :3], batch_bbox3d_targets[:, 3:6], out_dir_rad_pos], dim=-1)
            print('000:', decoded_bboxes3d[0], )
            print('111:', batch_bbox3d_targets[0], )
            print('222:', decoded_bboxes[batch_pos_binds][0])
            print('333:', batch_bbox_targets[0])
            # batch_diou3d = diou_3d_loss(decoded_bboxes3d_wodir, batch_bbox3d_targets, reduction='none')
            # batch_label_scores3d = torch.exp(-batch_diou3d * self.temperature_scores3d)
            batch_label_scores[batch_pos_binds] = batch_label_scores[batch_pos_binds]  # * batch_label_scores3d.detach()
            loss_qfl = self.loss_cls(
                out_cls_scores, (batch_labels, batch_label_scores), avg_factor=max(num_batch_pos, 1)
            )
            print('444:', out_cls_scores[batch_pos_binds][0])
            print('555:', batch_label_scores[batch_pos_binds][0])

            weight_targets = out_cls_scores[batch_pos_binds].detach().max(dim=1)[0]
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)
            loss_bbox = self.loss_bbox(
                decoded_bboxes[batch_pos_binds],
                batch_bbox_targets,
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )
            loss_dfl = self.loss_dfl(
                out_bbox_dist_preds[batch_pos_binds].reshape(-1, self.reg_max + 1),
                batch_dist_targets.reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor,
            )
            loss_gc3d = self.loss_gc3d(  # depth+2d sup
                decoded_gc3d.reshape(-1, 3),
                batch_bbox3d_targets[..., :3].reshape(-1, 3),
                # weight=weight_targets[:, None].expand(-1, 3).reshape(-1, 3),
                avg_factor=num_batch_pos  # 3.0 * bbox_avg_factor,
            )
            # loss_gc3d = out_bbox_dist_preds.sum() * 0
            # loss_offset2c3d = self.loss_offset2c3d(
            #     decoded_pc3d[batch_pos_binds].reshape(-1, 2),
            #     batch_kpts2d_targets[..., 0, :].reshape(-1, 2),
            #     weight=weight_targets[:, None].expand(-1, 2).reshape(-1, 2),
            #     avg_factor=2.0 * bbox_avg_factor,
            # )
            # loss_offset2c3d = out_bbox_dist_preds.sum() * 0
            loss_dfl_depth = self.loss_dfl(
                out_depth_dist_preds[batch_pos_binds].reshape(-1, self.depth_reg_max + 1),
                batch_depth_dist_targets.reshape(-1),
                # weight=weight_targets,
                avg_factor=num_batch_pos  # bbox_avg_factor,
            )
            # loss_depth = self.loss_depth(
            #
            # )
            loss_dim = self.loss_dim(
                out_dim_pos.reshape(-1, 3),
                batch_bbox3d_targets[..., 3:6].reshape(-1, 3),
                # weight=weight_targets[:, None].expand(-1, 3).reshape(-1, 3),
                avg_factor=num_batch_pos  # 3.0 * bbox_avg_factor,
            )
            # loss_dim = out_bbox_dist_preds.sum() * 0
            # batch_alpha_targets = -torch.atan2(batch_bbox3d_targets[..., 0], batch_bbox3d_targets[..., 2]) + \
            #                       batch_bbox3d_targets[..., 6]
            # batch_alpha_targets =
            loss_dir = self.loss_dir(
                out_dir_pos.reshape(-1, 2),
                batch_bbox3d_targets[..., 6].reshape(-1),
                # weight=weight_targets,
                avg_factor=num_batch_pos,
            )
            loss_dir_norm = (torch.norm(out_dir_pos.reshape(-1, 2), dim=-1) - 1).abs().mean()  # L2 norm
            mae_xyz = (decoded_bboxes3d[:, :3] - batch_bbox3d_targets[:, :3]).abs().mean()
            mae_dim = (decoded_bboxes3d[:, 3:6] - batch_bbox3d_targets[:, 3:6]).abs().mean()
            mae_roty = loss_dir / torch.pi * 180
            mae_poscls = out_cls_scores[batch_pos_binds].float().mean()

            # loss_dir = out_bbox_dist_preds.sum() * 0
            # loss_iou3d_wodir = self.loss_iou3d(
            #     decoded_bboxes3d_wodir,
            #     batch_bbox3d_targets,
            #     # weight=weight_targets,
            #     avg_factor=bbox_avg_factor,
            # )
            # # loss_iou3d = out_bbox_dist_preds.sum() * 0
            # loss_iou3d_c3d = self.loss_iou3d(
            #     decoded_bboxes3d_c3d,
            #     batch_bbox3d_targets,
            #     # weight=weight_targets,
            #     avg_factor=bbox_avg_factor,
            # )
            # loss_iou3d_depth = self.loss_iou3d(
            #     decoded_bboxes3d_depth,
            #     batch_bbox3d_targets,
            #     # weight=weight_targets,
            #     avg_factor=bbox_avg_factor,
            # )
            # loss_iou3d_dim = self.loss_iou3d(
            #     decoded_bboxes3d_dim,
            #     batch_bbox3d_targets,
            #     # weight=weight_targets,
            #     avg_factor=bbox_avg_factor,
            # )
        else:
            loss_qfl = self.loss_cls(
                out_cls_scores, (batch_labels, batch_label_scores), avg_factor=max(num_batch_pos, 1)
            )
            loss_bbox = out_bbox_dist_preds.sum() * 0
            loss_dfl = loss_bbox
            loss_gc3d = loss_bbox
            # loss_offset2c3d = out_bbox_dist_preds.sum() * 0
            loss_dfl_depth = loss_bbox
            loss_dim = loss_bbox
            loss_dir = loss_bbox
            # loss_iou3d_wodir = loss_bbox
            # loss_iou3d_c3d = loss_bbox
            # loss_iou3d_depth = loss_bbox
            # loss_iou3d_dim = loss_bbox
            loss_dir_norm = loss_bbox
            mae_xyz = loss_bbox
            mae_dim = loss_bbox
            mae_roty = loss_bbox
            mae_poscls = loss_bbox
            # loss_iou3d_dir = out_bbox_dist_preds.sum() * 0

        loss_sum = loss_qfl + loss_bbox + loss_dfl
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl, loss_dfl_depth=loss_dfl_depth,
                           loss_dir=loss_dir, loss_dir_norm=loss_dir_norm, loss_gc3d=loss_gc3d, loss_dim=loss_dim,
                           mae_xyz=mae_xyz, mae_dim=mae_dim, mae_roty=mae_roty,mae_poscls=mae_poscls)
        # print(self.canon_box_sizes)
        return loss_sum, loss_states
