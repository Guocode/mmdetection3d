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

from ...core import CameraInstance3DBoxes

INF = 1e8
EPS = 1e-12
PI = math.pi


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
        x = F.softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1), dim=-1)
        x = F.linear(x, self.project.type_as(x)).reshape(*shape[:-1], 4)
        return x


@HEADS.register_module()
class MonoGFocalV2SAICHead(AnchorFreeHead):
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
    Example:
        >>> self = MonoGFocalV2SAICHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 loss_dir=dict(type='DirCosineLoss'),
                 loss_dim=dict(type='SmoothL1Loss'),
                 loss_depth=dict(type='UncertainSmoothL1Loss'),
                 loss_offset2c3d=dict(type='SmoothL1Loss'),
                 reg_max=16,
                 reg_topk=4,
                 reg_channels=64,
                 add_mean=True,
                 aux_reg=True,
                 bbox3d_code_size=7,
                 **kwargs):

        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.total_dim = reg_topk
        self.aux_reg = aux_reg
        self.bbox3d_code_size = bbox3d_code_size
        if add_mean:
            self.total_dim += 1
        print('total dim = ', self.total_dim * 4)

        super(MonoGFocalV2SAICHead, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.integral = Integral(self.reg_max)
        self.loss_dfl = build_loss(loss_dfl)

        self.loss_dir = build_loss(loss_dir)
        self.loss_dim = build_loss(loss_dim)
        self.loss_depth = build_loss(loss_depth)
        self.loss_offset2c3d = build_loss(loss_offset2c3d)

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
            self.feat_channels, 2, 3, padding=1)
        self.offset2c3d_reg = nn.Conv2d(
            self.feat_channels, 2, 3, padding=1)
        # aux reg
        if self.aux_reg:
            self.offset2kpts_reg = nn.Conv2d(
                self.feat_channels, 8 * 2, 3, padding=1)
            self.c3dkptshm_reg = nn.Conv2d(
                self.feat_channels, 8 + 1, 3, padding=1)

        self.scales2d = nn.ModuleList(
            [Scale(1.0) for _ in self.strides])
        self.scales3d_depth = nn.ModuleList(
            [Scale(1.0) for _ in self.strides])
        self.scales3d_offset2c3d = nn.ModuleList(
            [Scale(1.0) for _ in self.strides])
        conf_vector = [nn.Conv2d(4 * self.total_dim, self.reg_channels, 1)]
        conf_vector += [self.relu]
        conf_vector += [nn.Conv2d(self.reg_channels, 1, 1), nn.Sigmoid()]

        self.reg_conf = nn.Sequential(*conf_vector)

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
        out_bbox_pred = []
        out_dir_pred = []
        out_dim_pred = []
        out_depth_pred = []
        out_offset2c3d_pred = []

        for lvlid, feat in enumerate(feats):
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(feat)
            for reg3d_conv in self.reg3d_convs:
                reg3d_feat = reg3d_conv(feat)
            bbox_pred = self.scales2d[lvlid](self.gfl_reg(reg_feat)).float()
            N, C, H, W = bbox_pred.size()
            prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
            prob_topk, _ = prob.topk(self.reg_topk, dim=2)

            if self.add_mean:
                stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)],
                                 dim=2)
            else:
                stat = prob_topk

            quality_score = self.reg_conf(stat.reshape(N, -1, H, W))
            # cls_score = self.gfl_cls(cls_feat).sigmoid() * quality_score
            cls_score = self.gfl_cls(cls_feat) * quality_score
            # 3D
            dir_pred = F.normalize(self.dir_reg(reg3d_feat), dim=1)
            dim_pred = self.dim_reg(reg3d_feat)
            depth_pred = self.depth_reg(reg3d_feat)
            depth_pred[:, 0, :, :] = self.scales3d_depth[lvlid](
                1. / (depth_pred[:, 0, :, :].sigmoid() + EPS) - 1).float()

            offset2c3d_pred = self.scales3d_offset2c3d[lvlid](self.offset2c3d_reg(reg3d_feat)).float()
            # aux
            if self.aux_reg:
                ofset2kpts_pred = self.offset2kpts_reg(reg3d_feat)
                c3dkptshm_pred = self.c3dkptshm_reg(reg3d_feat)

            out_cls_score.append(cls_score.flatten(start_dim=2))
            out_bbox_pred.append(bbox_pred.flatten(start_dim=2))
            out_dir_pred.append(dir_pred.flatten(start_dim=2))
            out_dim_pred.append(dim_pred.flatten(start_dim=2))
            out_depth_pred.append(depth_pred.flatten(start_dim=2))
            out_offset2c3d_pred.append(offset2c3d_pred.flatten(start_dim=2))

        out_cls_scores = torch.cat(out_cls_score, dim=2).permute(0, 2, 1)
        out_bbox_preds = torch.cat(out_bbox_pred, dim=2).permute(0, 2, 1)
        out_dir_preds = torch.cat(out_dir_pred, dim=2).permute(0, 2, 1)
        out_dim_preds = torch.cat(out_dim_pred, dim=2).permute(0, 2, 1)
        out_depth_preds = torch.cat(out_depth_pred, dim=2).permute(0, 2, 1)
        out_offset2c3d_preds = torch.cat(out_offset2c3d_pred, dim=2).permute(0, 2, 1)

        return out_cls_scores, out_bbox_preds, out_dir_preds, out_dim_preds, out_depth_preds, out_offset2c3d_preds  # multi_apply(self.forward_single, feats, self.scales)

    @force_fp32(apply_to=('out_cls_scores', 'out_bbox_preds', 'out_dir_preds', 'out_dim_preds', 'out_depth_preds',
                          'out_offset2c3d_preds'))
    def loss(self,
             out_cls_scores,
             out_bbox_preds,
             out_dir_preds,
             out_dim_preds,
             out_depth_preds,
             out_offset2c3d_preds,
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
        dis_preds = self.integral(out_bbox_preds) * multi_lvl_ltcenter_priors[..., 2, None]
        decoded_bboxes = distance2bbox(multi_lvl_ltcenter_priors[..., :2], dis_preds)
        decoded_c3d = multi_lvl_ltcenter_priors[..., :2] + out_offset2c3d_preds * multi_lvl_ltcenter_priors[
            ..., 2, None]
        batch_target_res = self.get_targets(out_cls_scores,
                                            multi_lvl_ltcenter_priors,
                                            decoded_bboxes,
                                            gt_bboxes,
                                            gt_labels, gt_bboxes3d, gt_kpts2d, gt_kpts2d_valid)
        loss, loss_states = self.get_loss_from_target(
            out_cls_scores,
            out_bbox_preds,
            out_dir_preds,
            out_dim_preds,
            out_depth_preds,
            decoded_c3d, decoded_bboxes, batch_target_res
        )
        return loss_states

    def get_bboxes(self, out_cls_scores, out_bbox_preds, out_dir_preds, out_dim_preds, out_depth_preds,
                   out_offset2c3d_preds, img_metas, thresh=0.4):
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
        # TODO transform 3dbox after 2dnms
        dis_preds = self.integral(out_bbox_preds) * multi_lvl_ltcenter_priors[..., 2, None]
        decoded_bboxes = distance2bbox(multi_lvl_ltcenter_priors[..., :2], dis_preds, max_shape=input_shape)
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
            bbox3d = self.decode_bbox3d(bbox3d, img_metas[i]['cam2img'])
            bbox3d = CameraInstance3DBoxes(bbox3d,box_dim=self.bbox3d_code_size, origin=(0.5, 0.5, 0.5))
            result_list.append([bbox, cls, bbox3d])
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
        if not isinstance(gt_bboxes3d, torch.Tensor):
            gt_bboxes3d = gt_bboxes3d.tensor.to(gt_bboxes.device)

        single_img_labels = center_priors.new_full(
            (num_priors,), self.num_classes, dtype=torch.long
        )
        single_img_label_scores = center_priors.new_zeros(single_img_labels.shape, dtype=torch.float)
        single_img_bbox_targets = torch.empty((0, 4))
        single_img_bbox3d_targets = torch.empty((0, 7))
        single_img_kpts2d_targets = torch.empty((0, 9, 2))
        single_img_kpts2d_valid_targets = torch.empty((0, 9))
        single_img_pos_inds = torch.empty((0))
        # No target
        if num_gts == 0:
            return single_img_labels, single_img_label_scores, single_img_bbox_targets, single_img_bbox3d_targets, \
                   single_img_kpts2d_targets, single_img_kpts2d_valid_targets, single_img_pos_inds

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
            # single_img_label_scores3d = pos_ious#3d iou
            single_img_bbox3d_targets = pos_gt_bboxes3d
            single_img_kpts2d_targets = pos_gt_kpts2d
            single_img_kpts2d_valid_targets = pos_gt_kpts2d_valid
            single_img_pos_inds = pos_inds
        return single_img_labels, single_img_label_scores, single_img_bbox_targets, single_img_dist_targets, \
               single_img_bbox3d_targets, single_img_kpts2d_targets, single_img_kpts2d_valid_targets, single_img_pos_inds

    def decode_bbox3d(self, bbox3d, cam2img):
        # 1. alpha -> roty
        bbox3d[:, 6] = (bbox3d[:, 6] + torch.atan2(bbox3d[:, 0] - cam2img[0, 2], cam2img[0, 0]) + PI) % (PI * 2) - PI
        # 2. 2d+depth -> 3d
        bbox3d[:, :3] = self.pts2Dto3D(bbox3d[:, :3], cam2img)
        return bbox3d

    @staticmethod
    def pts2Dto3D(points, cam2img):
        """
        Args:
            points (torch.Tensor): points in 2D images, [N, 3], \
                3 corresponds with x, y in the image and depth.
            cam2img (torch.Tensor): camera instrinsic, [3, 3]

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
        cam2img_mono[3, 3] = 1.
        # viewpad[:view.shape[0], :view.shape[1]] = points2D.new_tensor(view)
        inv_cam2img_mono = torch.inverse(cam2img_mono).permute(1, 0).to(points.device)
        # inv_cam2img_mono = torch.linalg.inv(cam2img_mono).permute(1, 0)
        # Do operation in homogenous coordinates.
        nbr_points = unnorm_points2D.shape[0]
        homo_points2D = torch.cat(
            [unnorm_points2D,
             points2D.new_ones((nbr_points, 1))], dim=1).view(-1, 1, 4)  #
        points3D = (homo_points2D @ inv_cam2img_mono).view(-1, 4)[:, :3]

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

    def get_loss_from_target(self, out_cls_scores, out_bbox_preds, out_dir_preds, out_dim_preds, out_depth_preds,
                             decoded_c3d, decoded_bboxes, target):
        device = out_cls_scores.device
        num_priors = out_cls_scores.size(1)
        batch_labels, batch_label_scores, batch_bbox_targets, batch_dist_targets, \
        batch_bbox3d_targets, batch_kpts2d_targets, batch_kpts2d_valid_targets, batch_pos_inds = target
        batch_pos_binds = torch.cat(
            [pos_inds + num_priors * batch_id for batch_id, pos_inds in enumerate(batch_pos_inds)])
        num_batch_pos = max(batch_pos_binds.size(0), 1.0)

        batch_labels = torch.cat(batch_labels, dim=0)
        batch_label_scores = torch.cat(batch_label_scores, dim=0)
        batch_bbox_targets = torch.cat(batch_bbox_targets, dim=0)
        batch_dist_targets = torch.cat(batch_dist_targets, dim=0)
        batch_bbox3d_targets = torch.cat(batch_bbox3d_targets, dim=0)
        batch_kpts2d_targets = torch.cat(batch_kpts2d_targets, dim=0)
        batch_kpts2d_valid_targets = torch.cat(batch_kpts2d_valid_targets, dim=0)

        out_cls_scores = out_cls_scores.reshape(-1, self.num_classes)
        out_bbox_preds = out_bbox_preds.reshape(-1, 4 * (self.reg_max + 1))
        out_dir_preds = out_dir_preds.reshape(-1, 2)
        out_dim_preds = out_dim_preds.reshape(-1, 3)
        out_depth_preds = out_depth_preds.reshape(-1, 2)
        decoded_c3d = decoded_c3d.reshape(-1, 2)

        decoded_bboxes = decoded_bboxes.reshape(-1, 4)

        loss_qfl = self.loss_cls(
            out_cls_scores, (batch_labels, batch_label_scores), avg_factor=num_batch_pos
        )

        if num_batch_pos > 0:
            weight_targets = out_cls_scores[batch_pos_binds].detach().max(dim=1)[0]
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

            loss_bbox = self.loss_bbox(
                decoded_bboxes[batch_pos_binds],
                batch_bbox_targets,
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )

            loss_dfl = self.loss_dfl(
                out_bbox_preds[batch_pos_binds].reshape(-1, self.reg_max + 1),
                batch_dist_targets.reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor,
            )
            batch_alpha_targets = -torch.atan2(batch_bbox3d_targets[..., 0], batch_bbox3d_targets[..., 2]) + \
                                  batch_bbox3d_targets[..., 6]
            loss_dir = self.loss_dir(
                out_dir_preds[batch_pos_binds].reshape(-1, 2),
                batch_alpha_targets.reshape(-1),
            )
            loss_dim = self.loss_dim(
                out_dim_preds[batch_pos_binds].reshape(-1, 3),
                batch_bbox3d_targets[..., 3:6].reshape(-1, 3),
            )
            loss_depth = self.loss_depth(
                out_depth_preds[batch_pos_binds].reshape(-1, 2),
                batch_bbox3d_targets[..., 2].reshape(-1)
            )
            loss_offset2c3d = self.loss_offset2c3d(  # TODO reproj back to 3d and calculate loss
                decoded_c3d[batch_pos_binds].reshape(-1, 2),
                batch_kpts2d_targets[..., 0, :].reshape(-1, 2)
            )
        else:
            loss_bbox = out_bbox_preds.sum() * 0
            loss_dfl = out_bbox_preds.sum() * 0
            loss_dir = out_bbox_preds.sum() * 0
            loss_dim = out_bbox_preds.sum() * 0
            loss_depth = out_bbox_preds.sum() * 0
            loss_offset2c3d = out_bbox_preds.sum() * 0
        loss_sum = loss_qfl + loss_bbox + loss_dfl
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl, loss_dir=loss_dir,
                           loss_dim=loss_dim, loss_depth=loss_depth, loss_offset2c3d=loss_offset2c3d)
        return loss_sum, loss_states
