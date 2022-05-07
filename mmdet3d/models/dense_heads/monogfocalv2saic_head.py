import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, bbox2distance, bbox_overlaps,
                        build_assigner, build_sampler, distance2bbox,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap,MlvlPointGenerator,bbox_xyxy_to_cxcywh)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.anchor_head import AnchorHead
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
import math
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
                 # stacked_convs=4,
                 # conv_cfg=None,
                 # strides=[8, 16, 32 ,64, 128],
                 # norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 reg_max=16,
                 reg_topk=4,
                 reg_channels=64,
                 add_mean=True,
                 **kwargs):

        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.total_dim = reg_topk
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

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
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
        # assert self.num_anchors == 1, 'anchor free version'
        self.gfl_cls = nn.Conv2d(
            self.feat_channels, self.num_classes, 3, padding=1)
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        # self.scales = nn.ModuleList(
        #     [Scale(1.0) for _ in self.strides])

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
        for m in self.reg_conf:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)
        normal_init(self.gfl_reg, std=0.01)

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
        out_bbox_pred = []
        out_cls_score = []
        # cls_feat = x
        # reg_feat = x
        for feat in feats:
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(feat)

            bbox_pred = self.gfl_reg(reg_feat).float()
            N, C, H, W = bbox_pred.size()
            prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
            prob_topk, _ = prob.topk(self.reg_topk, dim=2)

            if self.add_mean:
                stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)],
                                 dim=2)
            else:
                stat = prob_topk

            quality_score = self.reg_conf(stat.reshape(N, -1, H, W))
            cls_score = self.gfl_cls(cls_feat).sigmoid() * quality_score
            out_cls_score.append(cls_score.flatten(start_dim=2))
            out_bbox_pred.append(bbox_pred.flatten(start_dim=2))
        out_cls_score = torch.cat(out_cls_score,dim=2).permute(0, 2, 1)
        out_bbox_pred = torch.cat(out_bbox_pred,dim=2).permute(0, 2, 1)
        return out_cls_score,out_bbox_pred#multi_apply(self.forward_single, feats, self.scales)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
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
        batch_size = cls_scores.shape[0]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        multi_lvl_ltcenter_priors = self.get_multi_lvl_ltcenter_priors(batch_size,featmap_sizes,self.strides,dtype=torch.float32,device=cls_scores.device)
        # label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        dis_preds = self.integral(bbox_preds) * multi_lvl_ltcenter_priors[..., 2, None]
        decoded_bboxes = distance2bbox(multi_lvl_ltcenter_priors[..., :2], dis_preds)

        batch_assign_res = self.get_targets(cls_scores,
                multi_lvl_ltcenter_priors,
                decoded_bboxes,
                gt_bboxes,
                gt_labels,)
        loss, loss_states = self._get_loss_from_assign(
            cls_scores, bbox_preds, decoded_bboxes, batch_assign_res
        )
        return loss_states
    def get_bboxes(self, cls_preds, reg_preds, img_metas):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        device = cls_preds.device
        b = cls_preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = (input_height, input_width)

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width / stride))
            for stride in self.strides
        ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_lvl_ltcenter_priors(
                b,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        dis_preds = self.integral(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=0.05,
                nms_cfg=dict(type="nms", iou_threshold=0.6),
                max_num=100,
            )
            result_list.append(results)
        return result_list

    def get_targets(self,cls_preds,multi_lvl_ltcenter_priors,decoded_bboxes,gt_bboxes,gt_labels):
        batch_assign_res = multi_apply(
            self.target_assign_single_img,
            cls_preds.detach(),
            multi_lvl_ltcenter_priors,
            decoded_bboxes.detach(),
            gt_bboxes,
            gt_labels,
        )
        return batch_assign_res
    @torch.no_grad()
    def target_assign_single_img(
            self, cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels
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

        bbox_targets = torch.zeros_like(center_priors)
        dist_targets = torch.zeros_like(center_priors)
        labels = center_priors.new_full(
            (num_priors,), self.num_classes, dtype=torch.long
        )
        label_scores = center_priors.new_zeros(labels.shape, dtype=torch.float)
        # No target
        if num_gts == 0:
            return labels, label_scores, bbox_targets, dist_targets, 0

        assign_result = self.assigner.assign(
            cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels
        )
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes
        )
        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            dist_targets[pos_inds, :] = (
                    bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes)
                    / center_priors[pos_inds, None, 2]
            )
            dist_targets = dist_targets.clamp(min=0, max=self.reg_max - 0.1)
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
        return (
            labels,
            label_scores,
            bbox_targets,
            dist_targets,
            num_pos_per_img,
        )

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
    def get_multi_lvl_ltcenter_priors(self,batch_size,featmap_sizes,strides,dtype,device):
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
    def sample(self, assign_result, gt_bboxes):
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
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds
    def _get_loss_from_assign(self, cls_preds, reg_preds, decoded_bboxes, assign):
        device = cls_preds.device
        labels, label_scores, bbox_targets, dist_targets, num_pos = assign
        num_total_samples = max(
            reduce_mean(torch.tensor(sum(num_pos)).to(device)).item(), 1.0
        )

        labels = torch.cat(labels, dim=0)
        label_scores = torch.cat(label_scores, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        loss_qfl = self.loss_cls(
            cls_preds, (labels, label_scores), avg_factor=num_total_samples
        )

        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False
        ).squeeze(1)

        if len(pos_inds) > 0:
            weight_targets = cls_preds[pos_inds].detach().max(dim=1)[0]
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )

            dist_targets = torch.cat(dist_targets, dim=0)
            loss_dfl = self.loss_dfl(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),
                dist_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor,
            )
        else:
            loss_bbox = reg_preds.sum() * 0
            loss_dfl = reg_preds.sum() * 0

        loss = loss_qfl + loss_bbox + loss_dfl
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)
        return loss, loss_states

if __name__ == '__main__':
    import mmcv
    import torch
    from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule


    def tt_gfl_head_loss():
        """Tests yolox head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
            'pad_shape': (s, s, 3)
        }]
        train_cfg = mmcv.Config(
            dict(
                assigner=dict(
                    type='SimOTAAssigner',
                    center_radius=2.5,
                    candidate_topk=10,
                    iou_weight=3.0,
                    cls_weight=1.0))

        )
        strides = [4, 8]
        self = MonoGFocalV2SAICHead(
            num_classes=3,
            in_channels=256,
            train_cfg=train_cfg,
            strides=strides,
            loss_cls=dict(
                type='QualityFocalLoss',
                activated=True,
                beta=2.0,
                loss_weight=1.0),
            loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
            reg_max=16,
            reg_topk=4,
            reg_channels=64,
            add_mean=True,
            loss_bbox=dict(type='GIoULoss', loss_weight=2.0))

        feat = [
            torch.rand(2, 256, s // feat_size, s // feat_size)
            for feat_size in strides
        ]
        cls_scores, bbox_preds = self.forward(feat)

        # Test that empty ground truth encourages the network to predict background
        gt_bboxes = [
            torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
            torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874],[0.6667, 0.8757, 22.6326, 22.8874]]),
        ]
        gt_labels = [torch.LongTensor([2]),torch.LongTensor([2,1])]
        empty_gt_losses = self.loss(cls_scores, bbox_preds,
                                    gt_bboxes, gt_labels, img_metas)
        print(empty_gt_losses)
        # When there is no truth, the cls loss should be nonzero but there should
        # be no box loss.
        # empty_cls_loss = empty_gt_losses['loss_cls'].sum()
        # empty_box_loss = empty_gt_losses['loss_bbox'].sum()
        # print(empty_box_loss)
        # print(empty_cls_loss)


    tt_gfl_head_loss()
