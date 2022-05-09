import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from LibMTL.weighting import GradNorm

from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils.gaussian_target import (gaussian_radius, gen_gaussian_target)
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from mmcv.ops import boxes_iou_bev, diff_iou_rotated_3d

INF = 1e8
EPS = 1e-12
PI = np.pi


@HEADS.register_module()
class MonoConSAICHead(nn.Module):
    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 bbox3d_code_size=7,
                 num_kpt=9,
                 max_objs=30,
                 vector_regression_level=1,
                 pred_bbox2d=True,
                 loss_center_heatmap=None,
                 loss_wh=None,
                 loss_offset=None,
                 loss_center2kpt_offset=None,
                 loss_kpt_heatmap=None,
                 loss_kpt_heatmap_offset=None,
                 loss_dim=None,
                 loss_depth=None,
                 loss_alpha_dir=None,
                 loss_iou2d=None,
                 loss_iou3d=None,
                 scale_factor=4,
                 use_AN=False,
                 num_AN_affine=10,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        '''

        Args:
            in_channel:
            feat_channel:
            num_classes:
            bbox3d_code_size:
            num_kpt:
            max_objs:
            vector_regression_level:
            pred_bbox2d:
            loss_center_heatmap:
            loss_wh:
            loss_offset:
            loss_center2kpt_offset:
            loss_kpt_heatmap:
            loss_kpt_heatmap_offset:
            loss_dim:
            loss_depth:
            loss_alpha_dir:
            loss_iou2d:
            loss_iou3d:
            scale_factor: scale factor to original image plane
            use_AN:
            num_AN_affine:
            train_cfg:
            test_cfg:
            init_cfg:
        '''
        super(MonoConSAICHead, self).__init__()
        assert bbox3d_code_size >= 7
        self.num_classes = num_classes
        self.bbox3d_code_size = bbox3d_code_size
        self.pred_bbox2d = pred_bbox2d
        self.max_objs = max_objs
        self.num_kpt = num_kpt
        self.vector_regression_level = vector_regression_level

        self.use_AN = use_AN
        self.num_AN_affine = num_AN_affine
        self.norm = nn.BatchNorm2d
        self.scale_factor = scale_factor
        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)
        self.center2kpt_offset_head = self._build_head(in_channel, feat_channel, self.num_kpt * 2)
        self.kpt_heatmap_head = self._build_head(in_channel, feat_channel, self.num_kpt)
        self.kpt_heatmap_offset_head = self._build_head(in_channel, feat_channel, 2)
        self.dim_head = self._build_head(in_channel, feat_channel, 3)
        self.depth_head = self._build_head(in_channel, feat_channel, 2)
        # self.dir_head = self._build_head(in_channel, feat_channel, 2)
        self.alpha_dir_head = self._build_head(in_channel, feat_channel, 2)
        self.iou2d_aware_head = self._build_head(in_channel, feat_channel, 1)
        self.iou3d_aware_head = self._build_head(in_channel, feat_channel, 1)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.loss_center2kpt_offset = build_loss(loss_center2kpt_offset)
        self.loss_kpt_heatmap = build_loss(loss_kpt_heatmap)
        self.loss_kpt_heatmap_offset = build_loss(loss_kpt_heatmap_offset)
        self.loss_dim = build_loss(loss_dim)
        self.loss_iou2d = build_loss(loss_iou2d)
        self.loss_iou3d = build_loss(loss_iou3d)

        if 'Aware' in loss_dim['type']:
            self.dim_aware_in_loss = True
        else:
            self.dim_aware_in_loss = False
        self.loss_depth = build_loss(loss_depth)
        self.loss_dir = build_loss(loss_alpha_dir)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        # self.mtl_loss = GradNorm()#TODO mtl loss

    def _build_head(self, in_channel, feat_channel, out_channel):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            self._get_norm_layer(feat_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def _get_norm_layer(self, feat_channel):
        return self.norm(feat_channel, momentum=0.03, eps=0.001) if not self.use_AN else \
            self.norm(feat_channel, self.num_AN_affine, momentum=0.03, eps=0.001)

    def init_weights(self):
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)  # -2.19
        self.kpt_heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head, self.center2kpt_offset_head, self.depth_head,
                     self.kpt_heatmap_offset_head, self.dim_head, self.alpha_dir_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        center_heatmap_pred = torch.clamp(center_heatmap_pred, min=1e-4, max=1 - 1e-4)
        kpt_heatmap_pred = self.kpt_heatmap_head(feat).sigmoid()
        kpt_heatmap_pred = torch.clamp(kpt_heatmap_pred, min=1e-4, max=1 - 1e-4)

        offset_pred = self.offset_head(feat)
        kpt_heatmap_offset_pred = self.kpt_heatmap_offset_head(feat)

        wh_pred = self.wh_head(feat)
        center2kpt_offset_pred = self.center2kpt_offset_head(feat)
        dim_pred = self.dim_head(feat)
        depth_pred = self.depth_head(feat)
        depth_pred[:, 0, :, :] = 1. / (depth_pred[:, 0, :, :].sigmoid() + EPS) - 1

        alpha_dir_pred = F.normalize(self.alpha_dir_head(feat), dim=1)
        return center_heatmap_pred, wh_pred, offset_pred, center2kpt_offset_pred, kpt_heatmap_pred, \
               kpt_heatmap_offset_pred, dim_pred, alpha_dir_pred, depth_pred

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds', 'center2kpt_offset_preds',
                          'kpt_heatmap_preds', 'kpt_heatmap_offset_preds', 'dim_preds', 'alpha_dir_preds',
                           'depth_preds'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             center2kpt_offset_preds,
             kpt_heatmap_preds,
             kpt_heatmap_offset_preds,
             dim_preds,
             alpha_dir_preds,
             depth_preds,
             gt_bboxes,
             gt_labels,
             gt_bboxes_3d,
             gt_labels_3d,
             centers2d,
             depths,
             gt_kpts_2d,
             gt_kpts_valid_mask,
             img_metas,
             attr_labels=None,
             proposal_cfg=None,
             gt_bboxes_ignore=None):

        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) \
               == len(center2kpt_offset_preds) == len(kpt_heatmap_preds) == len(kpt_heatmap_offset_preds) \
               == len(dim_preds) == len(alpha_dir_preds) == 1

        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]
        center2kpt_offset_pred = center2kpt_offset_preds[0]
        kpt_heatmap_pred = kpt_heatmap_preds[0]
        kpt_heatmap_offset_pred = kpt_heatmap_offset_preds[0]
        dim_pred = dim_preds[0]
        alpha_dir_pred = alpha_dir_preds[0]
        depth_pred = depth_preds[0]

        batch_size,_,feat_h,feat_w = center_heatmap_pred.shape

        target_result = self.get_targets(gt_bboxes, gt_labels,
                                         gt_bboxes_3d,
                                         centers2d,
                                         depths,
                                         gt_kpts_2d,
                                         gt_kpts_valid_mask,
                                         center_heatmap_pred.shape,
                                         img_metas[0]['img_shape'],
                                         img_metas)

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        bbox_target = target_result['bbox_target']
        center2kpt_offset_target = target_result['center2kpt_offset_target']
        dim_target = target_result['dim_target']
        depth_target = target_result['depth_target']
        # alpha_cls_target = target_result['alpha_cls_target']
        alpha_dir_target = target_result['alpha_dir_target']
        bbox3d_target = target_result['bbox3d_target']
        kpt_heatmap_target = target_result['kpt_heatmap_target']
        kpt_heatmap_offset_target = target_result['kpt_heatmap_offset_target']

        indices = target_result['indices']
        indices_kpt = target_result['indices_kpt']

        mask_target = target_result['mask_target']
        mask_center2kpt_offset = target_result['mask_center2kpt_offset']
        mask_kpt_heatmap_offset = target_result['mask_kpt_heatmap_offset']

        cam2img = target_result['cam2img']

        # select desired preds and labels based on mask

        # 2d offset
        offset_pred = self.extract_input_from_tensor(offset_pred, indices, mask_target)
        offset_target = self.extract_target_from_tensor(offset_target, mask_target)
        # 2d size
        wh_pred = self.extract_input_from_tensor(wh_pred, indices, mask_target)
        wh_target = self.extract_target_from_tensor(wh_target, mask_target)
        # 2d bbox
        bbox_target = self.extract_target_from_tensor(bbox_target, mask_target)
        # 3d dim
        dim_pred = self.extract_input_from_tensor(dim_pred, indices, mask_target)
        dim_target = self.extract_target_from_tensor(dim_target, mask_target)
        # depth
        depth_pred = self.extract_input_from_tensor(depth_pred, indices, mask_target)
        depth_target = self.extract_target_from_tensor(depth_target, mask_target)
        # alpha dir
        alpha_dir_pred = self.extract_input_from_tensor(alpha_dir_pred, indices, mask_target)
        alpha_dir_target = self.extract_target_from_tensor(alpha_dir_target, mask_target)

        # bbox3d_target = self.extract_input_from_tensor(bbox3d_target,mask_target)
        # alpha_cls_target = self.extract_target_from_tensor(alpha_cls_target, mask_target).type(torch.long)
        # alpha_cls_onehot_target = alpha_cls_target.new_zeros([len(alpha_cls_target), self.num_alpha_bins]).scatter_(
        #     dim=1, index=alpha_cls_target.view(-1, 1), value=1)
        # alpha offset
        # alpha_offset_pred = self.extract_input_from_tensor(alpha_offset_pred, indices, mask_target)
        # alpha_offset_pred = torch.sum(alpha_offset_pred * alpha_cls_onehot_target, 1, keepdim=True)
        # alpha_offset_target = self.extract_target_from_tensor(alpha_offset_target, mask_target)
        # center2kpt offset
        center2kpt_offset_pred = self.extract_input_from_tensor(center2kpt_offset_pred,
                                                                indices, mask_target)  # B * (num_kpt * 2)
        center2kpt_offset_target = self.extract_target_from_tensor(center2kpt_offset_target, mask_target)
        mask_center2kpt_offset = self.extract_target_from_tensor(mask_center2kpt_offset, mask_target)
        # kpt heatmap offset
        kpt_heatmap_offset_pred = transpose_and_gather_feat(kpt_heatmap_offset_pred, indices_kpt)
        kpt_heatmap_offset_pred = kpt_heatmap_offset_pred.reshape(batch_size, self.max_objs, self.num_kpt * 2)
        kpt_heatmap_offset_pred = kpt_heatmap_offset_pred[mask_target]
        kpt_heatmap_offset_target = kpt_heatmap_offset_target[mask_target]
        mask_kpt_heatmap_offset = self.extract_target_from_tensor(mask_kpt_heatmap_offset, mask_target)

        # calculate loss
        loss_center_heatmap = self.loss_center_heatmap(center_heatmap_pred, center_heatmap_target)
        loss_kpt_heatmap = self.loss_kpt_heatmap(kpt_heatmap_pred, kpt_heatmap_target)

        loss_wh = self.loss_wh(wh_pred, wh_target)
        loss_offset = self.loss_offset(offset_pred, offset_target)

        self.single_level_center_priors = self.get_single_level_center_priors(1, center_heatmap_pred.shape[2:],
                                                                              1, center_heatmap_pred.dtype,
                                                                              center_heatmap_pred.device, flatten=False)
        single_level_center_priors = self.extract_input_from_tensor(
            self.single_level_center_priors.permute((0, 3, 1, 2)).repeat([batch_size,1,1,1]),
            indices % (feat_w*feat_h), mask_target)
        xs, ys = single_level_center_priors[..., 0], single_level_center_priors[..., 1]
        topk_xs = xs + offset_pred[..., 0]
        topk_ys = ys + offset_pred[..., 1]
        tl_x = (topk_xs - wh_pred[..., 0] / 2) * 4
        tl_y = (topk_ys - wh_pred[..., 1] / 2) * 4
        br_x = (topk_xs + wh_pred[..., 0] / 2) * 4
        br_y = (topk_ys + wh_pred[..., 1] / 2) * 4
        batch_bboxes_pred = torch.stack([tl_x, tl_y, br_x, br_y], dim=-1)
        loss_iou2d = self.loss_iou2d(batch_bboxes_pred, bbox_target)

        if self.dim_aware_in_loss:
            loss_dim = self.loss_dim(dim_pred, dim_target, dim_pred)
        else:
            loss_dim = self.loss_dim(dim_pred, dim_target)

        depth_pred, depth_log_variance = depth_pred[:, 0:1], depth_pred[:, 1:2]
        loss_depth = self.loss_depth(depth_pred, depth_target, depth_log_variance)

        center2kpt_offset_pred *= mask_center2kpt_offset
        loss_center2kpt_offset = self.loss_center2kpt_offset(center2kpt_offset_pred, center2kpt_offset_target,
                                                             avg_factor=(mask_center2kpt_offset.sum() + EPS))

        kpt_heatmap_offset_pred *= mask_kpt_heatmap_offset
        loss_kpt_heatmap_offset = self.loss_kpt_heatmap_offset(kpt_heatmap_offset_pred, kpt_heatmap_offset_target,
                                                               avg_factor=(mask_kpt_heatmap_offset.sum() + EPS))

        centers2d_pred = torch.stack([topk_xs,topk_ys],dim=-1)*4

        # 1. decode alpha
        alpha_pred = self.sincos2angle(alpha_dir_pred)  # self.decode_alpha_multibin(alpha_cls, alpha_offset)  # (b, k, 1)

        # 1.5 get projected center
        # center2d = batch_bboxes_pred  # (b, k, 2)

        # 2. recover rotY
        roty_pred = self.alpha2roty(centers2d_pred, alpha_pred, cam2img)  # (b, k, 3)

        # 2.5 recover box3d_center from center2d and depth
        center3d_pred = torch.cat([centers2d_pred, depth_pred], dim=-1)#.squeeze(0)
        center3d_pred = self.pts2Dto3D(center3d_pred, cam2img)#.unsqueeze(0)

        # 3. compose 3D box
        batch_bboxes_3d_pred = torch.cat([center3d_pred, dim_pred, roty_pred.unsqueeze(-1)], dim=-1)

        loss_iou3d = 1 - torch.mean(diff_iou_rotated_3d(batch_bboxes_3d_pred.view(1,-1,7), bbox3d_target.view(1,-1,7)))
        # if mask_target.sum() > 0:
        #     loss_alpha_cls = self.loss_alpha_cls(alpha_cls_pred, alpha_cls_onehot_target)
        # else:
        #     loss_alpha_cls = 0.0
        # loss_alpha_dir = self.loss_dir(alpha_dir_pred, alpha_dir_target)
        loss_alpha_dir = 1 - torch.mean(torch.sum(alpha_dir_pred * alpha_dir_target, dim=-1))  # neg cos
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_dim=loss_dim,
            loss_center2kpt_offset=loss_center2kpt_offset,
            loss_kpt_heatmap=loss_kpt_heatmap,
            loss_kpt_heatmap_offset=loss_kpt_heatmap_offset,
            loss_alpha_dir=loss_alpha_dir,
            loss_depth=loss_depth,
            loss_iou2d=loss_iou2d,
            loss_iou3d=loss_iou3d
        )

    def get_targets(self, gt_bboxes, gt_labels,
                    gt_bboxes_3d,
                    centers2d,
                    depths,
                    gt_kpts_2d,
                    gt_kpts_valid_mask,
                    feat_shape, img_shape,
                    img_metas):
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        calibs = []

        # objects as 2D center points
        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])

        # 2D attributes
        wh_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])
        offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])
        bbox_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 4])
        # 3D attributes
        dim_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 3])
        alpha_dir_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])
        # alpha_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        depth_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        bbox3d_target = []#gt_bboxes[-1].new_zeros([bs, self.max_objs, self.bbox3d_code_size])
        # 2D-3D kpt heatmap and offset
        center2kpt_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])
        kpt_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_kpt, feat_h, feat_w])
        kpt_heatmap_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])

        # indices
        indices = gt_bboxes[-1].new_zeros([bs, self.max_objs]).type(torch.cuda.LongTensor)
        indices_kpt = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt]).type(torch.cuda.LongTensor)

        # masks
        mask_target = gt_bboxes[-1].new_zeros([bs, self.max_objs])
        mask_center2kpt_offset = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])
        mask_kpt_heatmap_offset = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])

        cam2img = []
        for batch_id in range(bs):
            img_meta = img_metas[batch_id]

            gt_bbox = gt_bboxes[batch_id]
            if len(gt_bbox) < 1:
                continue
            gt_label = gt_labels[batch_id]
            gt_bbox_3d = gt_bboxes_3d[batch_id]
            if not isinstance(gt_bbox_3d, torch.Tensor):
                gt_bbox_3d = gt_bbox_3d.tensor.to(gt_bbox.device)

            depth = depths[batch_id]

            gt_kpt_2d = gt_kpts_2d[batch_id]
            gt_kpt_valid_mask = gt_kpts_valid_mask[batch_id]

            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) / 2 * width_ratio
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) / 2 * height_ratio
            gt_centers = torch.cat((center_x, center_y), dim=1)

            gt_kpt_2d = gt_kpt_2d.reshape(-1, self.num_kpt, 2)
            gt_kpt_2d[:, :, 0] *= width_ratio
            gt_kpt_2d[:, :, 1] *= height_ratio

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio

                dim = gt_bbox_3d[j][3: 6]
                alpha = -torch.atan2(gt_bbox_3d[j, 0], gt_bbox_3d[j, 2]) + gt_bbox_3d[j, 6]  # roty to alpha
                gt_kpt_2d_single = gt_kpt_2d[j]  # (9, 2)
                gt_kpt_valid_mask_single = gt_kpt_valid_mask[j]  # (9,)

                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                indices[batch_id, j] = cty_int * feat_w + ctx_int

                wh_target[batch_id, j, 0] = scale_box_w
                wh_target[batch_id, j, 1] = scale_box_h
                offset_target[batch_id, j, 0] = ctx - ctx_int
                offset_target[batch_id, j, 1] = cty - cty_int
                bbox_target[batch_id,j] = gt_bbox[j][0]

                dim_target[batch_id, j] = dim
                depth_target[batch_id, j] = depth[j]
                alpha_dir_target[batch_id, j] = self.angle2sincos(alpha)
                bbox3d_target.append(gt_bbox_3d[j])

                mask_target[batch_id, j] = 1

                for k in range(self.num_kpt):
                    kpt = gt_kpt_2d_single[k]
                    kptx_int, kpty_int = kpt.int()
                    kptx, kpty = kpt
                    vis_level = gt_kpt_valid_mask_single[k]
                    if vis_level < self.vector_regression_level:
                        continue

                    center2kpt_offset_target[batch_id, j, k * 2] = kptx - ctx_int
                    center2kpt_offset_target[batch_id, j, k * 2 + 1] = kpty - cty_int
                    mask_center2kpt_offset[batch_id, j, k * 2:k * 2 + 2] = 1

                    is_kpt_inside_image = (0 <= kptx_int < feat_w) and (0 <= kpty_int < feat_h)
                    if not is_kpt_inside_image:
                        continue

                    gen_gaussian_target(kpt_heatmap_target[batch_id, k],
                                        [kptx_int, kpty_int], radius)

                    kpt_index = kpty_int * feat_w + kptx_int
                    indices_kpt[batch_id, j, k] = kpt_index

                    kpt_heatmap_offset_target[batch_id, j, k * 2] = kptx - kptx_int
                    kpt_heatmap_offset_target[batch_id, j, k * 2 + 1] = kpty - kpty_int
                    mask_kpt_heatmap_offset[batch_id, j, k * 2:k * 2 + 2] = 1
                cam2img.append(torch.asarray(img_meta['cam2img'])[:3,:3])

        indices_kpt = indices_kpt.reshape(bs, -1)
        mask_target = mask_target.type(torch.bool)

        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            center2kpt_offset_target=center2kpt_offset_target,
            dim_target=dim_target,
            depth_target=depth_target,
            alpha_dir_target=alpha_dir_target,
            # alpha_offset_target=alpha_offset_target,
            kpt_heatmap_target=kpt_heatmap_target,
            kpt_heatmap_offset_target=kpt_heatmap_offset_target,
            indices=indices,
            indices_kpt=indices_kpt,
            mask_target=mask_target,
            mask_center2kpt_offset=mask_center2kpt_offset,
            mask_kpt_heatmap_offset=mask_kpt_heatmap_offset,
            bbox_target=bbox_target,
            bbox3d_target=torch.stack(bbox3d_target,dim=0),
            cam2img=torch.stack(cam2img,dim=0),
        )

        return target_result

    @staticmethod
    def extract_input_from_tensor(input, ind, mask):
        input = transpose_and_gather_feat(input, ind)
        return input[mask]

    @staticmethod
    def extract_target_from_tensor(target, mask):
        return target[mask]

    def angle2sincos(self, angle):
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

    def sincos2angle(self, sincos):
        return torch.atan2(sincos[..., 0], sincos[..., 1])

    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   center2kpt_offset_preds,
                   kpt_heatmap_preds,
                   kpt_heatmap_offset_preds,
                   dim_preds,
                   alpha_dir_preds,
                   depth_preds,
                   img_metas,
                   ):

        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) \
               == len(center2kpt_offset_preds) == len(kpt_heatmap_preds) == len(kpt_heatmap_offset_preds) \
               == len(dim_preds) == len(alpha_dir_preds) ==  1
        # scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
        box_type_3d = img_metas[0]['box_type_3d']

        batch_det_bboxes, batch_det_bboxes_3d, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            wh_preds[0],
            offset_preds[0],
            center2kpt_offset_preds[0],
            kpt_heatmap_preds[0],
            kpt_heatmap_offset_preds[0],
            dim_preds[0],
            alpha_dir_preds[0],
            depth_preds[0],
            img_metas[0]['img_shape'][:2],
            torch.stack([img_meta['cam2img'] for img_meta in img_metas],0),
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel,
            thresh=self.test_cfg.thresh)

        # if rescale:
        #     batch_det_bboxes[..., :4] *= self.scale_factor

        det_results = [
            [box_type_3d(batch_det_bboxes_3d,
                         box_dim=self.bbox3d_code_size, origin=(0.5, 0.5, 0.5)),
             batch_det_bboxes[:, -1],
             batch_labels,
             batch_det_bboxes,
             ]
        ]
        return det_results

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       center2kpt_offset_pred,
                       kpt_heatmap_pred,
                       kpt_heatmap_offset_pred,
                       dim_pred,
                       alpha_dir_pred,
                       depth_pred,
                       img_shape,
                       cam2img,
                       k=100,
                       kernel=3,
                       thresh=0.4):
        batch, _, feat_h, feat_w = center_heatmap_pred.shape
        assert batch == 1
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, ys, xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = xs + offset[..., 0]
        topk_ys = ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / feat_w)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / feat_h)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / feat_w)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / feat_h)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)  # (b, k, 5)

        # decode 3D prediction
        dim = transpose_and_gather_feat(dim_pred, batch_index)
        alpha_dir = transpose_and_gather_feat(alpha_dir_pred, batch_index)
        # alpha_offset = transpose_and_gather_feat(alpha_offset_pred, batch_index)
        depth_pred = transpose_and_gather_feat(depth_pred, batch_index)
        depth = depth_pred[:, :, 0:1]

        sigma = depth_pred[:, :, 1]
        sigma = torch.exp(-sigma)
        batch_bboxes[..., -1] *= sigma

        center2kpt_offset = transpose_and_gather_feat(center2kpt_offset_pred, batch_index)
        center2kpt_offset = center2kpt_offset.view(batch, k, self.num_kpt * 2)[..., :2]
        center2kpt_offset[..., ::2] += xs.view(batch, k, 1).expand(batch, k, 1)
        center2kpt_offset[..., 1::2] += ys.view(batch, k, 1).expand(batch, k, 1)

        center2d = center2kpt_offset

        center2d[..., ::2] *= (inp_w / feat_w)
        center2d[..., 1::2] *= (inp_h / feat_h)

        # 1. decode alpha
        alpha = self.sincos2angle(alpha_dir)  # self.decode_alpha_multibin(alpha_cls, alpha_offset)  # (b, k, 1)

        # 1.5 get projected center

        # 2. recover rotY
        rot_y = self.alpha2roty(center2d.view(batch*k,2), alpha.view(batch*k), cam2img.repeat(k,1,1)).view(batch,k,1) # (b, k, 3)

        # 2.5 recover box3d_center from center2d and depth
        center3d = torch.cat([center2d, depth], dim=-1).view(-1,3)
        center3d = self.pts2Dto3D(center3d, cam2img.repeat(k,1,1)).view(batch,k,3)

        # 3. compose 3D box
        batch_bboxes_3d = torch.cat([center3d, dim, rot_y], dim=-1)

        mask = batch_bboxes[..., -1] > thresh
        batch_bboxes = batch_bboxes[mask]
        batch_bboxes_3d = batch_bboxes_3d[mask]
        batch_topk_labels = batch_topk_labels[mask]

        return batch_bboxes, batch_bboxes_3d, batch_topk_labels

    def alpha2roty(self, centers, alpha, cam2img):
        '''

        Args:
            centers: [N,2]
            alpha: [N]
            cam2img: [N,3,3]

        Returns:
            roty: [N]
        '''
        device = centers.device
        cam2img = cam2img.to(device)#.unsqueeze(0)

        si = cam2img[:, 0, 0]
        rot_y = alpha + torch.atan2(centers[:, 0] - cam2img[:, 0, 2], si)

        while (rot_y > PI).any():
            rot_y[rot_y > PI] = rot_y[rot_y > PI] - 2 * PI
        while (rot_y < -PI).any():
            rot_y[rot_y < -PI] = rot_y[rot_y < -PI] + 2 * PI

        return rot_y

    @staticmethod
    def pts2Dto3D(points, cam2img):
        """
        Args:
            points (torch.Tensor): points in 2D images, [N, 3], \
                3 corresponds with x, y in the image and depth.
            cam2img (torch.Tensor): camera instrinsic, [N, 3, 3]

        Returns:
            torch.Tensor: points in 3D space. [N, 3], \
                3 corresponds with x, y, z in 3D space.
        """
        # assert view.shape[0] <= 4
        # assert view.shape[1] <= 4
        # assert points.shape[1] == 3
        cam2img = cam2img.to(points.device)
        points2D = points[:, :2]
        depths = points[:, 2:]#.view(-1, 1)
        unnorm_points2D = torch.cat([points2D * depths, depths], dim=1)

        cam2img_mono = torch.nn.functional.pad(cam2img,(0,1,0,1))#torch.eye(4, dtype=points2D.dtype, device=points2D.device)
        cam2img_mono[:,3,3]=1.
        # viewpad[:view.shape[0], :view.shape[1]] = points2D.new_tensor(view)
        inv_cam2img_mono = torch.inverse(cam2img_mono).permute(0, 2,1)

        # Do operation in homogenous coordinates.
        nbr_points = unnorm_points2D.shape[0]
        homo_points2D = torch.cat(
            [unnorm_points2D,
             points2D.new_ones((nbr_points, 1))], dim=1).view(-1,1,4)#
        points3D = torch.bmm(homo_points2D, inv_cam2img_mono).view(-1,4)[:, :3]

        return points3D

    @staticmethod
    def _topk_channel(scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        return topk_scores, topk_inds, topk_ys, topk_xs

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      centers2d=None,
                      depths=None,
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
                              gt_labels_3d, centers2d, depths, gt_kpts_2d, gt_kpts_valid_mask,
                              img_metas, attr_labels)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if proposal_cfg is None:
            return losses
        else:
            raise NotImplementedError

    @staticmethod
    def get_single_level_center_priors(
            batch_size, featmap_size, stride, dtype, device, flatten=True,
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
        if flatten:
            y = y.flatten()
            x = x.flatten()
        # strides = x.new_full((x.shape[0],), stride)
        proiors = torch.stack([x, y], dim=-1)
        return proiors.unsqueeze(0)  # .repeat(batch_size, 1, 1)


if __name__ == '__main__':
    from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes,
                                   DepthInstance3DBoxes, LiDARInstance3DBoxes)
    from mmdet3d.models.builder import build_head
    from mmcv import Config

    # a = MonoConSAICHead.get_single_level_center_priors(1,(4,4),1,torch.float32,'cpu')
    # print(a,a.shape)
    # b = get_topk_from_heatmap(torch.asarray([0,1,0],dtype=torch.float32).reshape(1,1,3,1),k=1)
    # print(b)

    head_cfg = dict(
        type='MonoConSAICHead',
        in_channel=64,
        feat_channel=64,
        num_classes=3,
        loss_center_heatmap=dict(type='CenterNetGaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_iou2d=dict(type='GIoULoss', loss_weight=1.0),
        loss_iou3d=dict(type='GIoULoss', loss_weight=1.0),
        loss_center2kpt_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_kpt_heatmap=dict(type='CenterNetGaussianFocalLoss', loss_weight=1.0),
        loss_kpt_heatmap_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_dim=dict(type='DimAwareL1Loss', loss_weight=1.0),
        loss_depth=dict(type='LaplacianAleatoricUncertaintyLoss', loss_weight=1.0),
        loss_alpha_dir=dict(type='L1Loss', loss_weight=1.0),
        # test_cfg=dict(topk=30, local_maximum_kernel=3, max_per_img=30, thresh=0.4)
    )
    self = build_head(head_cfg).cuda()
    self.test_cfg = Config(dict(topk=30, local_maximum_kernel=3, max_per_img=30, thresh=0.4))#.topk=30
    # self.test_cfg.thresh=0.4
    # self.test_cfg.local_maximum_kernel=3
    feats = torch.rand([1, 64, 96, 312], dtype=torch.float32).cuda()
    gt_bboxes = [
        torch.Tensor([[722.4446, 151.0021, 832.2931, 314.5869]]).cuda(),
    ]
    gt_bboxes_3d = [
        CameraInstance3DBoxes(torch.Tensor([[1.9047, 0.524511456489563, 8.4150, 1.2000, 1.8900, 0.4800, 0.0100]]), box_dim=7,origin=(0.5, 0.5, 0.5)),
    ]
    gt_labels = [torch.Tensor([1]).cuda().long() for i in range(1)]
    gt_labels_3d = gt_labels
    centers2d = [torch.Tensor([[775.7633, 231.4706]]).cuda(), ]
    depths = [
        torch.tensor([8.4150]).cuda(),
    ]
    kpts2d = [
        torch.tensor([[[775.7633, 231.4706],
                       [728.2701, 151.0556],
                       [722.4446, 153.0757],
                       [722.4446, 307.3683],
                       [728.2701, 314.4005],
                       [832.2931, 151.0021],
                       [820.6868, 153.0279],
                       [820.6868, 307.5345],
                       [832.2931, 314.5869]]]).cuda()
    ]
    kpts2d_valid = [
        torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32).cuda(),
    ]
    attr_labels = None

    img_metas = [
        dict(
            cam2img=torch.asarray([[7.070493e+02, 0.000000e+00, 6.160814e+02],
                                   [0.000000e+00, 7.070493e+02, 1.875066e+02],
                                   [0.000000e+00, 0.000000e+00, 1.000000e+00], ]),
            scale_factor=np.array([1., 1., 1., 1.], dtype=np.float32),
            pad_shape=[128, 128],
            trans_mat=np.array([[0.25, 0., 0.], [0., 0.25, 0], [0., 0., 1.]],
                               dtype=np.float32),
            affine_aug=False,
            img_shape=(384, 1248, 3),
            box_type_3d=CameraInstance3DBoxes) for i in range(1)
    ]
    ret_dict = self([feats])

    losses = self.loss(*ret_dict, gt_bboxes, gt_labels, gt_bboxes_3d,
                       gt_labels_3d, centers2d, depths, kpts2d, kpts2d_valid, img_metas)

    # test get_boxes
    results = self.get_bboxes(*ret_dict, img_metas)
    print(results)
    assert len(results) == 2
    assert len(results[0]) == 4
