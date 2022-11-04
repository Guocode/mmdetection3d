# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import torch
from mmcv.cnn import Conv2d, ConvModule, build_conv_layer
from mmcv.runner import force_fp32, BaseModule
from torch.nn import functional as F

from mmdet.core import multi_apply
from mmdet.core.bbox.builder import build_bbox_coder
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.gaussian_target import gather_feat
from mmdet.models import HEADS, build_loss, FPN

from mmdet3d.models.backbones.mobilenet_v2_bev import MobileNetV2
from mmdet3d.models.dense_heads import AnchorFreeMono3DHead
from mmdet3d.models.pv2bev.simplebev import SimpleBev

from mmdet3d.core import (draw_heatmap_gaussian, gaussian_radius,
                          CameraInstance3DBoxes)
from mmdet3d.models.utils import clip_sigmoid

import torch.nn as nn


class SeparateHead(BaseModule):
    """SeparateHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int, optional): Output channels.
            Default: 64.
        final_kernel (int, optional): Kernel size for the last conv layer.
            Default: 1.
        init_bias (float, optional): Initial bias. Default: -2.19.
        conv_cfg (dict, optional): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str, optional): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(SeparateHead, self).__init__(init_cfg=init_cfg)
        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

            if init_cfg is None:
                self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the
                    shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


@HEADS.register_module()
class SimpleBevMono3DHead(AnchorFreeMono3DHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 xyz_channel,
                 dim_channel,
                 ori_channel,
                 bbox_coder,
                 loss_cls=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=0.1),
                 loss_dir=None,
                 loss_attr=None,
                 norm_cfg=dict(type='BN2d'),
                 grav_ct=False,
                 intri_norm=False,
                 init_cfg=None,
                 **kwargs):
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dir=loss_dir,
            loss_attr=loss_attr,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.xyz_channel = xyz_channel
        self.dim_channel = dim_channel
        self.ori_channel = ori_channel
        self.grav_ct = grav_ct
        self.intri_norm = intri_norm
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.yconv = mmcv.cnn.ConvModule(2560, 256, 3, 1, 1, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        self.pv2bev = SimpleBev(xrange=(-50, 50), yrange=(-5, 1), zrange=(0., 100), xsize=128, ysize=10, zsize=128)
        self.bev_encoder = MobileNetV2(out_indices=(2, 4, 6))

        # self.bev_fpn = FPN(in_channels=[32, 96, 320], out_channels=256, num_outs=3)
        common_heads = dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2))
        heads = copy.deepcopy(common_heads)
        heads.update(dict(heatmap=(num_classes, 2)))
        self.task_heads = SeparateHead(init_bias=0., final_kernel=3, in_channels=320, heads=heads,
                                       num_cls=num_classes)
        self.register_buffer('canon_box_sizes', torch.Tensor(
            [
                [4.4367273, 1.62023594, 1.82311556],  # L,H,W
                [0.56116328, 1.62447786, 0.60935871],
                [1.6910185, 1.395831, 0.70756476],
                [10.24677962, 3.19091763, 2.67794529],
                [9.70456951, 3.34836208, 2.66604373]
            ])
                             )

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      **kwargs):
        outs = self(x, img_metas)
        loss_inputs = (gt_bboxes_3d, gt_labels) + (outs, img_metas)
        losses = self.loss(*loss_inputs)
        return losses

    def forward(self, feats, img_metas):
        # pv_feats = feats[0]
        pv_mlvl_feats = feats

        P = torch.zeros((feats[0].shape[0], 1, 4, 4))
        cam2imgs = torch.stack([m['cam2img'] for m in img_metas], 0)
        P[:, 0, :3, :3] = cam2imgs
        P[:, 0, 3, 3] = 1
        bev_mlvl_feats = []
        for pv_lvl_feats in pv_mlvl_feats:
            bev_lvl_feats = self.pv2bev(pv_lvl_feats.unsqueeze(1), P, img_metas[0]['img_shape'])
            bev_mlvl_feats.append(bev_lvl_feats)
        bev_mlvl_feats = torch.stack(bev_mlvl_feats).sum(0)
        # bev_feats = self.pv2bev(pv_feats.unsqueeze(1), P, img_metas[0]['img_shape'])
        bev_feats = bev_mlvl_feats.permute(0, 1, 3, 2, 4)
        B, C, Y, X, Z = bev_feats.shape
        bev_feats = bev_feats.reshape(B, C * Y, X, Z)
        bev_feats = self.yconv(bev_feats)
        bev_feats = self.bev_encoder(bev_feats)
        # bbox_pred [dx,dy,z, h,w,l,sinroty,cosroty]
        ret_dicts = self.task_heads(bev_feats[-1])
        return ret_dicts

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, img_metas):
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d, img_metas)
        heatmaps = torch.stack(heatmaps)
        anno_boxes = torch.stack(anno_boxes)
        inds = torch.stack(inds)
        masks = torch.stack(masks)

        return heatmaps, anno_boxes, inds, masks

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, img_metas):
        device = gt_labels_3d.device
        if self.grav_ct:
            gt_bboxes_3d = torch.cat(
                (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
                dim=1).to(device)
        else:
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(device)
        if self.intri_norm:
            gt_bboxes_3d[:3] = gt_bboxes_3d[:3] / img_metas['cam2img'][0, 0] * 466
        max_objs = self.train_cfg['max_objs']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        feature_map_size = grid_size[:2]
        draw_gaussian = draw_heatmap_gaussian
        heatmap = gt_bboxes_3d.new_zeros(
            (self.num_classes, feature_map_size[1],
             feature_map_size[0]))

        anno_box = gt_bboxes_3d.new_zeros((max_objs, 8),
                                          dtype=torch.float32)

        ind = -gt_labels_3d.new_ones((max_objs), dtype=torch.int64)
        mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

        num_objs = min(gt_bboxes_3d.shape[0], max_objs)
        norm_gt_bboxes_3d = self.pv2bev.norm2bev(gt_bboxes_3d[:, :3], 'xyz')

        for k in range(num_objs):
            cls_id = gt_labels_3d[k]
            width = gt_bboxes_3d[k][3] / (self.pv2bev.xrange[1] - self.pv2bev.xrange[0]) * self.pv2bev.xsize
            length = gt_bboxes_3d[k][5] / (self.pv2bev.xrange[1] - self.pv2bev.xrange[0]) * self.pv2bev.xsize

            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width),
                    min_overlap=self.train_cfg['gaussian_overlap'])
                radius = max(self.train_cfg['min_radius'], int(radius))

                # be really careful for the coordinate system of
                # your box annotation.
                # x, y, z = gt_bboxes_3d[k][0], gt_bboxes_3d[k][1], gt_bboxes_3d[k][2]
                enc_x, _, enc_z = norm_gt_bboxes_3d[k][0], norm_gt_bboxes_3d[k][1], norm_gt_bboxes_3d[k][2]
                center = torch.tensor([enc_z, enc_x],
                                      dtype=torch.float32,
                                      device=device)
                center_int = center.to(torch.int32)

                # throw out not in range objects to avoid out of array
                # area when creating the heatmap
                if not (0 <= center_int[0] < feature_map_size[0]
                        and 0 <= center_int[1] < feature_map_size[1]):
                    continue

                draw_gaussian(heatmap[cls_id.long()], center_int, radius)

                new_idx = k
                z, x = center_int[0], center_int[1]
                y = gt_bboxes_3d[k][1]
                assert (x * feature_map_size[0] + z <
                        feature_map_size[0] * feature_map_size[1])

                ind[new_idx] = x * feature_map_size[0] + z
                mask[new_idx] = 1
                # TODO: support other outdoor dataset
                rot = gt_bboxes_3d[k][6]
                box_dim = gt_bboxes_3d[k][3:6]
                box_dim = box_dim / self.canon_box_sizes[gt_labels_3d[k]]
                anno_box[new_idx] = torch.cat([
                    center - torch.tensor([z, x], device=device),
                    y.unsqueeze(0),
                    box_dim,
                    torch.sin(rot).unsqueeze(0),
                    torch.cos(rot).unsqueeze(0),
                ])
        return heatmap, anno_box, ind, mask

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        rets = []
        batch_size = preds_dicts['heatmap'].shape[0]
        batch_heatmap = preds_dicts['heatmap'].sigmoid()
        batch_reg = preds_dicts['reg']
        batch_hei = preds_dicts['height']
        batch_dim = preds_dicts['dim'].tanh().exp()
        batch_roty = torch.atan2(preds_dicts['rot'][:, 0].unsqueeze(1), preds_dicts['rot'][:, 1].unsqueeze(1))
        batch, cat, _, _ = batch_heatmap.size()
        fmap_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(batch_heatmap)
        keep = (batch_heatmap - fmap_max).float() + 1e-9
        keep = nn.ReLU()(keep)
        keep = keep * 1e9
        batch_heatmap = batch_heatmap * keep
        for i in range(batch_size):
            inds = torch.where(batch_heatmap[i] > self.test_cfg['score_threshold'])
            reg = batch_reg[i, :, inds[1], inds[2]]
            reg[0] += inds[2]  # Z
            reg[1] += inds[1]  # X
            reg = self.pv2bev.unorm(reg.T, 'zx').T
            hei = batch_hei[i, :, inds[1], inds[2]]
            dim = batch_dim[i, :, inds[1], inds[2]] * self.canon_box_sizes[inds[0]].T
            roty = batch_roty[i, :, inds[1], inds[2]]

            boxes3d = torch.cat([reg[1:], hei, reg[:1], dim, roty], dim=0).T
            if self.intri_norm:
                boxes3d[:, :3] = boxes3d[:, :3] / 466 * img_metas[i]['cam2img'][0, 0]
            scores = batch_heatmap[i][inds]
            labels = inds[0]
            if self.grav_ct:
                boxes3d = CameraInstance3DBoxes(boxes3d, origin=(0.5, 0.5, 0.5))
            else:
                boxes3d = CameraInstance3DBoxes(boxes3d, origin=(0.5, 1.0, 0.5))
            boxes_scores = torch.cat([scores.new_zeros((scores.shape[0], 4)), scores.unsqueeze(-1)], dim=-1)
            rets.append([boxes_scores, boxes3d, labels])
        return rets

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, img_metas, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, img_metas)
        loss_dict = dict()
        # heatmap focal loss
        preds_dicts['heatmap'] = clip_sigmoid(preds_dicts['heatmap'])
        num_pos = heatmaps.eq(1).float().sum().item()
        loss_heatmap = self.loss_cls(
            preds_dicts['heatmap'],
            heatmaps,
            avg_factor=max(num_pos, 1))
        target_box = anno_boxes
        # reconstruct the anno_box from multiple reg heads
        preds_dicts['anno_box'] = torch.cat(
            (preds_dicts['reg'], preds_dicts['height'],
             preds_dicts['dim'].tanh().exp(), preds_dicts['rot'],
             ),
            dim=1)

        # Regression loss for dimension, offset, height, rotation
        ind = inds
        pred = preds_dicts['anno_box'].permute(0, 2, 3, 1).contiguous()
        pred = pred.view(pred.size(0), -1, pred.size(3))
        mask = (ind != -1).unsqueeze(2).expand_as(target_box).float()
        ind[ind == -1] = 0
        pred = gather_feat(pred, ind)
        isnotnan = (~torch.isnan(target_box)).float()
        mask *= isnotnan

        code_weights = self.train_cfg.get('code_weights', None)
        bbox_weights = mask * mask.new_tensor(code_weights)

        loss_bbox = self.loss_bbox(
            pred, target_box, bbox_weights, avg_factor=(num_pos + 1e-4))
        loss_dict['loss_heatmap'] = loss_heatmap
        loss_dict['loss_bbox'] = loss_bbox
        return loss_dict

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor, optional): Mask of the feature map with the
                shape of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat


if __name__ == '__main__':
    SimpleBevMono3DHead()






























import torch
import torch.nn.functional as F

EPS = 1e-8


class SimpleBev(torch.nn.Module):
    def __init__(self, xrange=(-50, 50), yrange=(-50, 50), zrange=(-1, 5), xsize=200, ysize=200, zsize=20):
        super().__init__()
        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange
        self.xsize = xsize
        self.ysize = ysize
        self.zsize = zsize
        x = torch.arange(xrange[0], xrange[1], (xrange[1] - xrange[0]) / (xsize), dtype=torch.float32)
        y = torch.arange(yrange[0], yrange[1], (yrange[1] - yrange[0]) / (ysize), dtype=torch.float32)
        z = torch.arange(zrange[0], zrange[1], (zrange[1] - zrange[0]) / (zsize), dtype=torch.float32)
        self.bev_xyz = torch.stack(torch.meshgrid(x, y, z), -1)  # [X,Y,Z,3]
        self.bev_xyzh = torch.cat([self.bev_xyz, torch.ones(self.bev_xyz.shape[:-1] + (1,))], -1)
        print()

    def forward(self, pv_feats, pv_params, img_shape):
        """

        Args:
            pv_feats: B, N, C, H, W
            pv_params: B, N, 4, 4
            img_shape:
        """
        B, N, C, H, W = pv_feats.shape
        img_h, img_w, _ = img_shape
        pix_uvdh = torch.einsum('bnhw,xyzw->bnxyzh', pv_params.to(pv_feats.device), self.bev_xyzh.to(pv_feats.device))
        _, _, X, Y, Z, _ = pix_uvdh.shape
        pix_uv = pix_uvdh[..., :2] / pix_uvdh[..., 2:3]
        pix_uv[..., 0:1] = (pix_uv[..., 0:1] - img_w / 2) / (img_w / 2)  # convert to pytorch coordinate
        pix_uv[..., 1:2] = (pix_uv[..., 1:2] - img_h / 2) / (img_h / 2)
        pix_uvh = torch.cat([pix_uv, pix_uv.new_zeros(pix_uv.shape[:-1] + (1,))], -1)
        valid = (pix_uv[..., 0] > -1) * (pix_uv[..., 0] < 1) * (pix_uv[..., 1] > -1) * (pix_uv[..., 1] < 1) * (
                pix_uvdh[..., 2] > EPS)
        bev_feats = F.grid_sample(pv_feats.reshape(B * N, C, 1, H, W), pix_uvh.reshape(B * N, X, Y, Z, 3),
                                  align_corners=False, padding_mode='border')  # 1,h,w
        bev_feats = bev_feats.reshape(B, N, C, X, Y, Z) * valid.reshape(B, N, 1, X, Y, Z)
        mask = (bev_feats != 0).float()
        numer = torch.sum(bev_feats, dim=1)
        denom = EPS + torch.sum(mask, dim=1)

        bev_feats = numer / denom
        return bev_feats

    def norm2bev(self, co, dims):
        assert co.shape[-1] == len(dims)
        norm_co = torch.clone(co)
        for i, d in enumerate(dims):
            if d == 'x':
                norm_co[..., i] = (co[..., i] - self.xrange[0]) / (self.xrange[1] - self.xrange[0]) * self.xsize
            elif d == 'y':
                norm_co[..., i] = (co[..., i] - self.yrange[0]) / (self.yrange[1] - self.yrange[0]) * self.ysize
            elif d == 'z':
                norm_co[..., i] = (co[..., i] - self.zrange[0]) / (self.zrange[1] - self.zrange[0]) * self.zsize
            else:
                raise Exception("unknown dims")
        return norm_co

    def unorm(self, norm_co, dims):
        assert norm_co.shape[-1] == len(dims)
        co = torch.clone(norm_co)
        for i, d in enumerate(dims):
            if d == 'x':
                co[..., i] = norm_co[..., i] * (self.xrange[1] - self.xrange[0]) / self.xsize + self.xrange[0]
            elif d == 'y':
                co[..., i] = norm_co[..., i] * (self.yrange[1] - self.yrange[0]) / self.ysize + self.yrange[0]
            elif d == 'z':
                co[..., i] = norm_co[..., i] * (self.zrange[1] - self.zrange[0]) / self.zsize + self.zrange[0]
            else:
                raise Exception("unknown dims")
        return co


if __name__ == '__main__':
    a = SimpleBev()
    a1 = torch.rand(2, 2, 3, 5, 5)
    a2 = torch.eye(4)[None, None, ...].repeat(2, 2, 1, 1)
    bev_feats, valid = a(a1, a2, {"img_shape": (100, 100)})
    print(bev_feats.size())
