# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule, normal_init, bias_init_with_prob
from mmdet.core import multi_apply
from torch import nn
from torch.nn import functional as F
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.gaussian_target import (get_local_maximum,
                                                get_topk_from_heatmap,
                                                transpose_and_gather_feat)

from mmdet3d.core import CameraInstance3DBoxes
from ..bev_transformer.conv_bev_transformer import ConvBEVTransformer
from ..builder import HEADS, build_loss
from .base_mono3d_dense_head import BaseMono3DDenseHead


@HEADS.register_module()
class BEVMono3DHead(BaseMono3DDenseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 loss_cls=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=0.1),
                 loss_dir=dict(type='DirCosineLoss'),
                 loss_iou3d=dict(type='DIOU3DLoss'),
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 bev_size=(128, 128),
                 group_reg_dims=(2, 1, 3, 2),  # offset:2,y:1,dim:3,roty:2
                 xrange=(-30, 30),
                 zrange=(0, 60),
                 init_cfg=None,
                 **kwargs):
        super(BEVMono3DHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.group_reg_dims = group_reg_dims
        self.bev_size = bev_size  # (h,w) <->(z,x)
        self.conv_bias = conv_bias
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.xrange = xrange
        self.zrange = zrange
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_dir = build_loss(loss_dir)
        self.loss_iou3d = build_loss(loss_iou3d)
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()
        self.bev_pool = nn.AdaptiveAvgPool2d(self.bev_size)
        self.bev_transformer = ConvBEVTransformer(in_channels=self.in_channels,
                                                  out_channels=self.in_channels,
                                                  feat_channels=128,
                                                  stacked_bev_convs_num=3,
                                                  kernel_s=9,
                                                  )

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def init_weights(self):
        """Initialize weights of the head.

        We currently still use the customized defined init_weights because the
        default init of DCN triggered by the init_cfg will init
        conv_offset.weight, which mistakenly affects the training stability.
        """
        for modules in [self.cls_convs, self.reg_convs, self.conv_cls_prev]:
            for m in modules:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        for conv_reg_prev in self.conv_reg_prevs:
            if conv_reg_prev is None:
                continue
            for m in conv_reg_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        if self.use_direction_classifier:
            for m in self.conv_dir_cls_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        if self.pred_attrs:
            for m in self.conv_attr_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        for conv_reg in self.conv_regs:
            normal_init(conv_reg, std=0.01)
        if self.use_direction_classifier:
            normal_init(self.conv_dir_cls, std=0.01, bias=bias_cls)
        if self.pred_attrs:
            normal_init(self.conv_attr, std=0.01, bias=bias_cls)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * bbox_code_size.
        """

        feats_singlelvl = feats[-1]
        feats_singlelvl = self.bev_pool(feats_singlelvl)
        bev_feats = self.bev_transformer(feats_singlelvl)
        cls_feat = bev_feats
        reg_feat = bev_feats

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = torch.sigmoid(self.conv_cls(cls_feat))

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        regs_out = self.conv_regs(reg_feat)
        b, c, h, w = cls_score.shape
        cls_score = cls_score.permute(0, 2, 3, 1).view(b, h * w, c)
        b, c, h, w = regs_out.shape
        regs_out = regs_out.permute(0, 2, 3, 1).view(b, h * w, c)
        offset_out = regs_out[..., :2]
        centery_out = regs_out[..., 2:3]
        dim_out = regs_out[..., 3:6]
        dir_out = F.normalize(regs_out[..., 6:],dim=-1)
        return cls_score, offset_out, centery_out, dim_out, dir_out

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_regs = nn.Conv2d(self.feat_channels, sum(self.group_reg_dims), 1)

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, rescale=None):
        """Generate bboxes from bbox head predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level.
            bbox_preds (list[Tensor]): Box regression for each scale.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[tuple[:obj:`CameraInstance3DBoxes`, Tensor, Tensor, None]]:
                Each item in result_list is 4-tuple.
        """
        assert len(cls_scores) == len(bbox_preds) == 1
        cam2imgs = torch.stack([
            cls_scores[0].new_tensor(img_meta['cam2img'])
            for img_meta in img_metas
        ])
        trans_mats = torch.stack([
            cls_scores[0].new_tensor(img_meta['trans_mat'])
            for img_meta in img_metas
        ])
        batch_bboxes, batch_scores, batch_topk_labels = self.decode_heatmap(
            cls_scores[0],
            bbox_preds[0],
            img_metas,
            cam2imgs=cam2imgs,
            trans_mats=trans_mats,
            topk=100,
            kernel=3)

        result_list = []
        for img_id in range(len(img_metas)):
            bboxes = batch_bboxes[img_id]
            scores = batch_scores[img_id]
            labels = batch_topk_labels[img_id]

            keep_idx = scores > 0.25
            bboxes = bboxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]

            bboxes = img_metas[img_id]['box_type_3d'](
                bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
            attrs = None
            result_list.append((bboxes, scores, labels, attrs))

        return result_list

    def decode_heatmap(self,
                       cls_score,
                       reg_pred,
                       img_metas,
                       cam2imgs,
                       trans_mats,
                       topk=100,
                       kernel=3):
        """Transform outputs into detections raw bbox predictions.

        Args:
            class_score (Tensor): Center predict heatmap,
                shape (B, num_classes, H, W).
            reg_pred (Tensor): Box regression map.
                shape (B, channel, H , W).
            img_metas (List[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cam2imgs (Tensor): Camera intrinsic matrixs.
                shape (B, 4, 4)
            trans_mats (Tensor): Transformation matrix from original image
                to feature map.
                shape: (batch, 3, 3)
            topk (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of SMOKEHead, containing
               the following Tensors:
              - batch_bboxes (Tensor): Coords of each 3D box.
                    shape (B, k, 7)
              - batch_scores (Tensor): Scores of each 3D box.
                    shape (B, k)
              - batch_topk_labels (Tensor): Categories of each 3D box.
                    shape (B, k)
        """
        img_h, img_w = img_metas[0]['pad_shape'][:2]
        bs, _, feat_h, feat_w = cls_score.shape

        center_heatmap_pred = get_local_maximum(cls_score, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=topk)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        regression = transpose_and_gather_feat(reg_pred, batch_index)
        regression = regression.view(-1, 8)

        points = torch.cat([topk_xs.view(-1, 1),
                            topk_ys.view(-1, 1).float()],
                           dim=1)
        locations, dimensions, orientations = self.bbox_coder.decode(
            regression, points, batch_topk_labels, cam2imgs, trans_mats)

        batch_bboxes = torch.cat((locations, dimensions, orientations), dim=1)
        batch_bboxes = batch_bboxes.view(bs, -1, self.bbox_code_size)
        return batch_bboxes, batch_scores, batch_topk_labels

    def get_predictions(self, labels3d, centers2d, gt_locations, gt_dimensions,
                        gt_orientations, indices, img_metas, pred_reg):
        """Prepare predictions for computing loss.

        Args:
            labels3d (Tensor): Labels of each 3D box.
                shape (B, max_objs, )
            centers2d (Tensor): Coords of each projected 3D box
                center on image. shape (B * max_objs, 2)
            gt_locations (Tensor): Coords of each 3D box's location.
                shape (B * max_objs, 3)
            gt_dimensions (Tensor): Dimensions of each 3D box.
                shape (N, 3)
            gt_orientations (Tensor): Orientation(yaw) of each 3D box.
                shape (N, 1)
            indices (Tensor): Indices of the existence of the 3D box.
                shape (B * max_objs, )
            img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            pre_reg (Tensor): Box regression map.
                shape (B, channel, H , W).

        Returns:
            dict: the dict has components below:
            - bbox3d_yaws (:obj:`CameraInstance3DBoxes`):
                bbox calculated using pred orientations.
            - bbox3d_dims (:obj:`CameraInstance3DBoxes`):
                bbox calculated using pred dimensions.
            - bbox3d_locs (:obj:`CameraInstance3DBoxes`):
                bbox calculated using pred locations.
        """
        batch, channel = pred_reg.shape[0], pred_reg.shape[1]
        w = pred_reg.shape[3]
        cam2imgs = torch.stack([
            gt_locations.new_tensor(img_meta['cam2img'])
            for img_meta in img_metas
        ])
        trans_mats = torch.stack([
            gt_locations.new_tensor(img_meta['trans_mat'])
            for img_meta in img_metas
        ])
        centers2d_inds = centers2d[:, 1] * w + centers2d[:, 0]
        centers2d_inds = centers2d_inds.view(batch, -1)
        pred_regression = transpose_and_gather_feat(pred_reg, centers2d_inds)
        pred_regression_pois = pred_regression.view(-1, channel)
        locations, dimensions, orientations = self.bbox_coder.decode(
            pred_regression_pois, centers2d, labels3d, cam2imgs, trans_mats,
            gt_locations)

        locations, dimensions, orientations = locations[indices], dimensions[
            indices], orientations[indices]

        locations[:, 1] += dimensions[:, 1] / 2

        gt_locations = gt_locations[indices]

        assert len(locations) == len(gt_locations)
        assert len(dimensions) == len(gt_dimensions)
        assert len(orientations) == len(gt_orientations)
        bbox3d_yaws = self.bbox_coder.encode(gt_locations, gt_dimensions,
                                             orientations, img_metas)
        bbox3d_dims = self.bbox_coder.encode(gt_locations, dimensions,
                                             gt_orientations, img_metas)
        bbox3d_locs = self.bbox_coder.encode(locations, gt_dimensions,
                                             gt_orientations, img_metas)

        pred_bboxes = dict(ori=bbox3d_yaws, dim=bbox3d_dims, loc=bbox3d_locs)

        return pred_bboxes

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, img_metas):
        batch_targets_res = multi_apply(
            self.get_targets_single_img,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas
        )
        return batch_targets_res

    @torch.no_grad()
    def get_targets_single_img(self, gt_bboxes_3d, gt_labels_3d, img_metas):
        """Get training targets for batch images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gt,).
            gt_bboxes_3d (list[:obj:`CameraInstance3DBoxes`]): 3D Ground
                truth bboxes of each image,
                shape (num_gt, bbox_code_size).
            gt_labels_3d (list[Tensor]): 3D Ground truth labels of each
                box, shape (num_gt,).
            centers2d (list[Tensor]): Projected 3D centers onto 2D image,
                shape (num_gt, 2).
            feat_shape (tuple[int]): Feature map shape with value,
                shape (B, _, H, W).
            img_shape (tuple[int]): Image shape in [h, w] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple[Tensor, dict]: The Tensor value is the targets of
                center heatmap, the dict has components below:
              - gt_centers2d (Tensor): Coords of each projected 3D box
                    center on image. shape (B * max_objs, 2)
              - gt_labels3d (Tensor): Labels of each 3D box.
                    shape (B, max_objs, )
              - indices (Tensor): Indices of the existence of the 3D box.
                    shape (B * max_objs, )
              - affine_indices (Tensor): Indices of the affine of the 3D box.
                    shape (N, )
              - gt_locs (Tensor): Coords of each 3D box's location.
                    shape (N, 3)
              - gt_dims (Tensor): Dimensions of each 3D box.
                    shape (N, 3)
              - gt_yaws (Tensor): Orientation(yaw) of each 3D box.
                    shape (N, 1)
              - gt_cors (Tensor): Coords of the corners of each 3D box.
                    shape (N, 8, 3)
        """
        feat_h, feat_w = self.bev_size
        if isinstance(gt_bboxes_3d, CameraInstance3DBoxes):  # target bbox3d with gravity center and alpha
            gt_bboxes_3d = gt_bboxes_3d.tensor
        deivce = gt_bboxes_3d[-1].device
        single_img_cls_score_target = gt_bboxes_3d[-1].new_zeros(
            [1, self.num_classes, feat_h, feat_w])
        single_img_offset_target = []
        single_img_centery_target = []
        single_img_dim_target = []
        single_img_roty_target = []
        single_img_pos_inds = []
        for i in range(len(gt_bboxes_3d)):
            bev_center = gt_bboxes_3d[i, [0, 2]]
            bev_center[0] = bev_center[0] / (self.xrange[1] - self.xrange[0]) * self.bev_size[1]
            bev_center[1] = self.bev_size[0] - bev_center[1] / (self.zrange[1] - self.zrange[0]) * self.bev_size[0]

            bev_dim_x, bev_dim_z = gt_bboxes_3d[i, [3, 5]]
            bev_dim_x = bev_dim_x / (self.xrange[1] - self.xrange[0]) * self.bev_size[1]
            bev_dim_z = bev_dim_z / (self.zrange[1] - self.zrange[0]) * self.bev_size[0]
            bev_center_int = bev_center.int()
            bev_center_x_int, bev_center_z_int = bev_center_int
            radius = gaussian_radius([bev_dim_z, bev_dim_x], min_overlap=0.0)
            radius = max(0, int(radius * 2))
            gen_gaussian_target(single_img_cls_score_target[0, gt_labels_3d[i]],
                                bev_center.int(), radius)
            single_img_offset_target.append(bev_center - bev_center_int)
            single_img_centery_target.append(gt_bboxes_3d[i, 1:2])
            single_img_dim_target.append(gt_bboxes_3d[i, 3:6])
            single_img_roty_target.append(gt_bboxes_3d[i, 6:])
            single_img_pos_inds.append(bev_center_z_int * feat_w + bev_center_x_int)
        single_img_cls_score_target = single_img_cls_score_target.permute(0, 2, 3, 1).view(1, feat_h * feat_w,
                                                                                           self.num_classes)
        single_img_bboxes3d_target = gt_bboxes_3d.view(-1, 7)
        single_img_offset_target = torch.stack(single_img_offset_target).view(-1, 2).to(deivce)
        single_img_centery_target = torch.stack(single_img_centery_target).view(-1, 1).to(deivce)
        single_img_dim_target = torch.stack(single_img_dim_target).view(-1, 3).to(deivce)
        single_img_roty_target = torch.stack(single_img_roty_target).view(-1, 1).to(deivce)
        single_img_pos_inds = torch.stack(single_img_pos_inds).view(-1).to(deivce).long()
        return single_img_cls_score_target, single_img_bboxes3d_target, single_img_offset_target, single_img_centery_target, \
               single_img_dim_target, single_img_roty_target, single_img_pos_inds

    def loss(self, cls_score, offset_out, centery_out, dim_out, dir_out,
             gt_bboxes_3d,
             gt_labels_3d,
             img_metas):
        bev_ltcenter_priors = self.get_bev_ltcenter_priors(cls_score.size(0), self.bev_size, self.xrange, self.zrange,
                                                           cls_score.dtype, cls_score.device)
        decode_center_out = bev_ltcenter_priors + offset_out
        batch_target = self.get_targets(gt_bboxes_3d,
                                        gt_labels_3d,
                                        img_metas)
        loss_dict = self.get_loss_from_target(cls_score, offset_out, centery_out, dim_out, dir_out, batch_target,
                                              decode_center_out)

        return loss_dict

    def get_bev_ltcenter_priors(
            self, batch_size, bev_size, xrange, zrange, dtype, device
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
        h, w = bev_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) / w * (xrange[1] - xrange[0])
        z_range = (1 - (torch.arange(h, dtype=dtype, device=device)) / h) * (zrange[1] - zrange[0]) + zrange[0]
        z, x = torch.meshgrid(z_range, x_range)
        z = z.flatten()
        x = x.flatten()
        # strides = x.new_full((x.shape[0],), stride)
        proiors = torch.stack([x, z], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)

    def get_loss_from_target(self, cls_score, offset_out, centery_out, dim_out, dir_out, batch_target,
                             decode_centerxz_out):
        device = cls_score.device
        num_priors = cls_score.size(1)
        batch_size = cls_score.size(0)
        batch_cls_score_target, batch_bboxes3d_target, batch_offset_target, batch_centery_target, \
        batch_dim_target, batch_roty_target, batch_pos_inds = batch_target
        batch_pos_binds = torch.cat(
            [pos_inds + num_priors * batch_id for batch_id, pos_inds in enumerate(batch_pos_inds)])
        batch_pos_bids = torch.cat(
            [torch.full_like(pos_inds, batch_id) for batch_id, pos_inds in enumerate(batch_pos_inds)])
        num_batch_pos = batch_pos_binds.size(0)
        batch_cls_score_target = torch.cat(batch_cls_score_target, dim=0)
        batch_bboxes3d_target = torch.cat(batch_bboxes3d_target, dim=0)
        batch_offset_target = torch.cat(batch_offset_target, dim=0)
        batch_centery_target = torch.cat(batch_centery_target, dim=0)
        batch_dim_target = torch.cat(batch_dim_target, dim=0)
        batch_roty_target = torch.cat(batch_roty_target, dim=0)
        cls_score = cls_score.view(-1, self.num_classes)
        offset_out = offset_out.view(-1, 2)
        centery_out = centery_out.view(-1, 1)
        dim_out = dim_out.view(-1, 3)
        dir_out = dir_out.view(-1, 2)
        decode_centerxz_out = decode_centerxz_out.view(-1,2)
        pos_centerxz_out = decode_centerxz_out[batch_pos_binds]
        pos_offset_out = offset_out[batch_pos_binds]
        pos_centery_out = centery_out[batch_pos_binds]
        pos_dim_out = dim_out[batch_pos_binds]
        pos_dir_out = dir_out[batch_pos_binds]
        decoded_bboxes3d_offset = torch.cat(
            [pos_centerxz_out[:, :1], batch_bboxes3d_target[:, 1:2], pos_centerxz_out[:, 1:],
             batch_bboxes3d_target[:, 3:6], batch_bboxes3d_target[:, 6:]], dim=-1)
        decoded_bboxes3d_centery = torch.cat(
            [batch_bboxes3d_target[:, :1], pos_centery_out, batch_bboxes3d_target[:, 2:3],
             batch_bboxes3d_target[:, 3:6], batch_bboxes3d_target[:, 6:]], dim=-1)
        decoded_bboxes3d_dim = torch.cat(
            [batch_bboxes3d_target[:, :3], pos_dim_out,
             batch_bboxes3d_target[:, 6:]], dim=-1)

        loss_cls = self.loss_cls(
            cls_score, batch_cls_score_target.view(-1, self.num_classes), avg_factor=max(num_batch_pos, 1))
        loss_offset = self.loss_iou3d(
            decoded_bboxes3d_offset,
            batch_bboxes3d_target,
        )
        loss_centery = self.loss_iou3d(
            decoded_bboxes3d_centery,
            batch_bboxes3d_target,
        )
        loss_dim = self.loss_iou3d(
            decoded_bboxes3d_dim,
            batch_bboxes3d_target,
        )
        loss_roty = self.loss_dir(
            pos_dir_out.reshape(-1, 2),
            batch_roty_target.reshape(-1),
        )

        return dict(
            loss_cls=loss_cls,
            loss_offset=loss_offset,
            loss_centery=loss_centery,
            loss_dim=loss_dim,
            loss_roty=loss_roty
        )


if __name__ == '__main__':
    head = BEVMono3DHead()
    x = torch.rand((1, 128, 128, 128), dtype=torch.float32)
    bev_outs = head(x)
    print(bev_outs.shape)
