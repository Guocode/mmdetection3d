# Copyright (c) OpenMMLab. All rights reserved.
import math
import random

import torch
from mmcv.cnn import ConvModule, normal_init, bias_init_with_prob, NonLocal2d
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, build_assigner, build_sampler
from mmdet.models.utils import SinePositionalEncoding
from torch import nn
from torch.nn import functional as F

from mmdet3d.core import CameraInstance3DBoxes
from ..bev_transformer.conv_bev_transformer import ConvBEVTransformer
from mmdet.models.builder import HEADS, build_loss
from mmdet3d.models.dense_heads.base_mono3d_dense_head import BaseMono3DDenseHead


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)
        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5,
                                                                                 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class ResConvModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResConvModule, self).__init__()
        self.conv_module = ConvModule(*args, **kwargs)

    def forward(self, x):
        res = x
        x = self.conv_module(x)
        if res.size(1) == x.size(1):
            return x + res
        return x


class Channel_Attention_Module_FC(nn.Module):
    def __init__(self, channels, ratio):
        super(Channel_Attention_Module_FC, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        avg_x = self.avg_pooling(x).view(b, c)
        max_x = self.max_pooling(x).view(b, c)
        v = self.fc_layers(avg_x) + self.fc_layers(max_x)
        v = self.sigmoid(v).view(b, c, 1, 1)
        return x * v


class Spatial_Attention_Module(nn.Module):
    def __init__(self, k: int):
        super(Spatial_Attention_Module, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        # In order to keep the size of the front and rear images consistent
        # with calculate, k = 1 + 2p, k denote kernel_size, and p denote padding number
        # so, when p = 1 -> k = 3; p = 2 -> k = 5; p = 3 -> k = 7, it works. when p = 4 -> k = 9, it is too big to use in network
        assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        self.conv = nn.Conv2d(2, 1, kernel_size=(k, k), stride=(1, 1), padding=((k - 1) // 2, (k - 1) // 2),
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # compress the C channel to 1 and keep the dimensions
        avg_x = self.avg_pooling(x, dim=1, keepdim=True)
        max_x, _ = self.max_pooling(x, dim=1, keepdim=True)
        v = self.conv(torch.cat((max_x, avg_x), dim=1))
        v = self.sigmoid(v)
        return x * v


class Channel_Attention_Module_Conv(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(Channel_Attention_Module_Conv, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pooling(x)
        max_x = self.max_pooling(x)
        avg_out = self.conv(avg_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(avg_out + max_out)
        return x * v


class CBAMBlock(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock, self).__init__()
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = Channel_Attention_Module_FC(channels=channels, ratio=ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = Channel_Attention_Module_Conv(channels=channels, gamma=gamma, b=b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k=spatial_attention_kernel_size)

    def forward(self, x):
        x = self.channel_attention_block(x)
        x = self.spatial_attention_block(x)
        return x


@HEADS.register_module()
class BEVDNDETRMono3DHead(BaseMono3DDenseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=128,
                 stacked_convs=4,
                 # loss_cls=dict(type='CrossEntropyLoss', loss_weight=5.0),
                 loss_cls=dict(type='FocalLoss', loss_weight=1.0),
                 loss_dir=dict(type='DirCosineLoss', loss_weight=0.1),
                 # loss_reg=dict(type='L1Loss'),
                 loss_reg=dict(type='SmoothL1Loss', beta=0.05),
                 loss_iou3d=dict(type='DIOU3DLoss'),
                 bg_cls_weight=0.1,
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 group_reg_dims=(3, 3, 2),  # xyz:3,dim:3,rotysincos:2
                 xrange=(-40., 40.),
                 yrange=(-0.94, 3.00),
                 zrange=(0.3, 80.),
                 bev_prior_size=(64, 64),
                 num_query_sqrt=15,
                 bev_transformer_chn=None,
                 bev_transformer_kernel_s=9,
                 bev_transformer_convs_num=3,
                 dn_train=False,
                 dense_assign=False,
                 in_h=12,
                 in_w=40,
                 z_size=16,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(BEVDNDETRMono3DHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes + 1
        self.reg_out_channels = sum(group_reg_dims)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.group_reg_dims = group_reg_dims
        self.conv_bias = conv_bias
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.xrange = (-zrange[1] * 640 / 710, zrange[1] * 640 / 710)  # xrange
        self.xrange_min = (-zrange[0] * 640 / 710, zrange[0] * 640 / 710)  # xrange

        # self.yrange = yrange
        self.yrange = (-zrange[1] * 192 / 710, zrange[1] * 192 / 710)
        self.yrange_min = (-zrange[0] * 192 / 710, zrange[0] * 192 / 710)
        self.in_h = in_h
        self.in_w = in_w
        self.z_size = z_size

        self.zrange = zrange
        self.register_buffer('xyzrange',
                             torch.asarray([self.xrange, self.yrange, zrange, self.xrange_min, self.yrange_min],
                                           dtype=torch.float32))  # (3,2)
        self.bev_prior_size = bev_prior_size
        # self.num_query_sqrt = num_query_sqrt
        # self.num_query = num_query_sqrt ** 2
        self.bev_transformer_chn = feat_channels if bev_transformer_chn is None else bev_transformer_chn
        self.bev_transformer_convs_num = bev_transformer_convs_num
        self.bev_transformer_kernel_s = bev_transformer_kernel_s
        self.bg_cls_weight = bg_cls_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)
        self.loss_dir = build_loss(loss_dir)
        self.loss_iou3d = build_loss(loss_iou3d)
        self._init_layers()
        self._init_canon_box_sizes()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.dn_train = dn_train
        self.dense_assign = False

        x_prior = torch.arange(0, self.in_w, device='cuda') / (self.in_w - 1) * 2 - 1
        y_prior = torch.zeros((self.z_size, self.in_w), device='cuda')
        z_prior = torch.arange(0, self.z_size, device='cuda') / (self.z_size - 1) * 2 - 1
        z_prior, x_prior = torch.meshgrid(z_prior, x_prior)
        xyz_prior = torch.stack([x_prior, y_prior, z_prior.flip(0)])[None]
        # self.z_prior = torch.arange(0,12,device='cuda')/12*2-1
        self.register_buffer('xyz_prior', xyz_prior.reshape(1, 3, -1).permute(0, 2, 1))  # (1,num_prior,3)

        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # self.dsassigner = build_assigner(self.train_cfg.dsassigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.bev_transformer = ConvBEVTransformer(in_channels=self.in_channels,
                                                  out_channels=self.feat_channels,
                                                  feat_channels=self.bev_transformer_chn,
                                                  stacked_bev_convs_num=self.bev_transformer_convs_num,
                                                  # kernel_s=self.bev_transformer_kernel_s,
                                                  in_h=self.in_h,
                                                  z_size=self.z_size,
                                                  )
        self._init_clsreg_convs()
        self.ffn = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Conv2d(self.feat_channels, self.cls_out_channels + self.reg_out_channels, 1),
        )

        self.init_weights()

    def _init_clsreg_convs(self):
        """Initialize classification conv layers of the head."""
        clsreg_convs = nn.ModuleList()
        chout = self.feat_channels
        for i in range(self.stacked_convs):
            chin = chout
            chout = self.feat_channels
            conv_cfg = self.conv_cfg
            # clsreg_convs.append(
            #     NonLocal2d(chout, reduction=4, sub_sample=True)
            # )
            clsreg_convs.append(
                ResConvModule(
                    chin,
                    chout,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias),
            )
        self.clsreg_convs = nn.Sequential(*clsreg_convs)

    def init_weights(self):
        """Initialize weights of the head.

        We currently still use the customized defined init_weights because the
        default init of DCN triggered by the init_cfg will init
        conv_offset.weight, which mistakenly affects the training stability.
        """
        for conv_cls in self.clsreg_convs:
            normal_init(conv_cls, std=0.01)
        for m in self.ffn:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01, bias=0.0)

    def _init_canon_box_sizes(self):
        canon = torch.asarray(
            [[1.61876949, 3.89154523, 1.52969237],  # Car
             [0.62806586, 0.82038497, 1.76784787],  # Pedestrian
             [0.56898187, 1.77149234, 1.7237099],  # Cyclist
             [1.9134491, 5.15499603, 2.18998422],  # Van
             [2.61168401, 9.22692319, 3.36492722],  # Truck
             [0.5390196, 1.08098042, 1.28392158],  # Person_sitting
             [2.36044838, 15.56991038, 3.5289238],  # Tram
             [1.24489164, 2.51495357, 1.61402478],  # Misc
             ])[:, [1, 2, 0]]
        self.register_buffer('canon_box_sizes',canon)
        # self.canon_box_sizes = nn.Parameter(canon)

    def normxyz2xyzrange(self, normxyz):
        '''
        :param xyznorm: [...,3] @(-1,1)
        :return: xyzr: [...,3]
        '''
        shape = normxyz.shape
        normxyz = normxyz.reshape(-1, 3) * 0.5 + 0.5
        zr = normxyz[:, 2:3] * (self.xyzrange[2:3, 1] - self.xyzrange[2:3, 0])[None] + self.xyzrange[2:3, 0][None]
        cyr = self.xyzrange[4][None] + (self.xyzrange[1] - self.xyzrange[4])[None] * normxyz[:, 2:3]
        # yr = normxyz[:, 1:2] * (self.xyzrange[1:2, 1] - self.xyzrange[1:2, 0])[None] + self.xyzrange[1:2, 0][None]
        yr = normxyz[:, 1:2] * (cyr[:, 1:2] - cyr[:, 0:1]) + cyr[:, 0:1]
        cxr = self.xyzrange[3][None] + (self.xyzrange[0] - self.xyzrange[3])[None] * normxyz[:, 2:3]
        xr = normxyz[:, 0:1] * (cxr[:, 1:2] - cxr[:, 0:1]) + cxr[:, 0:1]
        xyzr = torch.cat([xr, yr, zr], dim=-1)
        xyzr = xyzr.reshape(*shape)
        return xyzr  # xyz * (self.xyzrange[:3, 1] - self.xyzrange[:3, 0])[None, None] + self.xyzrange[:3, 0][None, None]

    def xyzrange2normxyz(self, xyzrange):
        '''
        :param xyzrange: [...,3]
        :return: xyzn: [...,3] @(-1,1)
        '''
        shape = xyzrange.shape
        xyzrange = xyzrange.reshape(-1, 3)
        zn = (xyzrange[:, 2:3] - self.xyzrange[2:3, 0][None]) / (self.xyzrange[2:3, 1] - self.xyzrange[2:3, 0])[None]
        cyr = self.xyzrange[4][None] + (self.xyzrange[1] - self.xyzrange[4])[None] * zn

        # yn = (xyzrange[:, 1:2] - self.xyzrange[1:2, 0][None]) / (self.xyzrange[1:2, 1] - self.xyzrange[1:2, 0])[None]
        yn = (xyzrange[:, 1:2] - cyr[:, 0:1]) / (cyr[:, 1:2] - cyr[:, 0:1])

        cxr = self.xyzrange[3][None] + (self.xyzrange[0] - self.xyzrange[3])[None] * zn
        xn = (xyzrange[:, 0:1] - cxr[:, 0:1]) / (cxr[:, 1:2] - cxr[:, 0:1])
        xyzn = torch.cat([xn, yn, zn], dim=-1) * 2 - 1.
        xyzn = xyzn.reshape(*shape)
        return xyzn  # (xyzrange - self.xyzrange[:, 0][None]) / (self.xyzrange[:, 1] - self.xyzrange[:, 0])[None]

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes=None,
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
        loss_inputs = outs + (gt_bboxes_3d,
                              gt_labels,
                              img_metas)
        losses = self.loss(*loss_inputs)

        if proposal_cfg is None:
            return losses
        else:
            raise NotImplementedError

    def forward(self, feats):
        feats_singlelvl = feats[0]  #
        bev_feat = self.bev_transformer(feats_singlelvl)
        b, c, h, w = bev_feat.shape
        for clsreg_conv in self.clsreg_convs:
            bev_feat = clsreg_conv(bev_feat)
        bev_out = self.ffn(bev_feat)
        cls_reg_out = bev_out.view(b, -1, h * w).permute(0, 2, 1)
        cls_out, reg_out = cls_reg_out[..., :self.cls_out_channels], cls_reg_out[..., self.cls_out_channels:]
        b, nq, c = cls_out.shape
        cls_out = cls_out.view(b, nq, self.num_classes + 1)
        b, nq, c = reg_out.shape
        reg_out = reg_out.view(b, nq, sum(self.group_reg_dims))
        reg_xyz_out = reg_out[..., 0:3]  # torch.tanh(reg_out[..., 0:3])
        print('xyz_offset',reg_xyz_out[...,[0,2]].abs().mean())
        reg_xyz_out = torch.cat([torch.tanh(reg_xyz_out[..., 0:1])/self.in_w*2, reg_xyz_out[..., 1:2], torch.tanh(reg_xyz_out[..., 2:3])/self.z_size*2],
                                dim=-1) + self.xyz_prior  # (-1,1)
        reg_dim_out = torch.tanh(reg_out[..., 3:6])  # .clamp(-1.,1.)  # (0,2)
        reg_roty_out = reg_out[..., 6:8]  # -1,1
        regs_out = torch.cat([reg_xyz_out, reg_dim_out, reg_roty_out], dim=-1)
        return cls_out, regs_out

    @force_fp32(apply_to=('cls_out', 'reg_out'))
    def get_bboxes(self, cls_out, reg_out, img_metas, rescale=None):
        result_list = []
        for img_id in range(len(img_metas)):
            scores, labels = cls_out[img_id].sigmoid().max(-1)
            pos_idx = (scores > 0.35) * (labels < self.num_classes)
            scores = scores[pos_idx]
            labels = labels[pos_idx]

            reg_pos = reg_out[img_id][pos_idx]
            # bboxes3d_dim = torch.exp(reg_pos[:, 3:6]) * self.canon_box_sizes[labels]
            # bboxes3d = torch.cat(
            #     [self.normxyz2xyzrange(reg_pos[:, :3]), bboxes3d_dim, torch.atan2(reg_pos[:, 6:7], reg_pos[:, 7:8])],
            #     dim=-1)
            bboxes3d = self.get_gt_from_reg(reg_pos, labels,keep_dir=False)
            bboxes3d = CameraInstance3DBoxes(bboxes3d, box_dim=7, origin=(0.5, 0.5, 0.5))
            bboxes2d = torch.cat(
                [torch.zeros((scores.size(0), 4), dtype=scores.dtype, device=scores.device), scores.view(-1, 1)],
                dim=-1)
            result_list.append((bboxes2d, bboxes3d, labels))
        return result_list

    def alpha2roty(self, alpha, xyz):
        '''

        :param alpha: N*2
        :param xyz: N*3
        :return:
        '''
        sinroty = alpha[:, 0:1] * xyz[:, 2:3] / (torch.sqrt(xyz[:, 0:1] ** 2 + xyz[:, 2:3] ** 2) + 1e-8) + \
                  alpha[:, 1:2] * xyz[:, 0:1] / (torch.sqrt(xyz[:, 0:1] ** 2 + xyz[:, 2:3] ** 2) + 1e-8)
        cosroty = alpha[:, 1:2] * xyz[:, 2:3] / (torch.sqrt(xyz[:, 0:1] ** 2 + xyz[:, 2:3] ** 2) + 1e-8) - \
                  alpha[:, 0:1] * xyz[:, 0:1] / (torch.sqrt(xyz[:, 0:1] ** 2 + xyz[:, 2:3] ** 2) + 1e-8)
        return torch.cat([sinroty, cosroty], dim=-1)

    def get_targets(self, cls_out, reg_out, decode_bboxes3d, gt_bboxes_3d, gt_labels_3d, img_metas):
        batch_targets_res = multi_apply(
            self.get_targets_single_img,
            cls_out,
            reg_out,
            decode_bboxes3d,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas
        )
        return batch_targets_res  # + batch_dntargets_res

    @torch.no_grad()
    def get_targets_single_img(self, cls_out, reg_out, decode_bboxes3d, gt_bboxes_3d, gt_labels_3d, img_meta):
        # assigner and sampler
        device = cls_out.device
        num_query = cls_out.size(0)
        if isinstance(gt_bboxes_3d, CameraInstance3DBoxes):  # target bbox3d with gravity center and alpha
            gt_bboxes_3d = torch.cat([gt_bboxes_3d.bottom_center, gt_bboxes_3d.dims, gt_bboxes_3d.yaw[..., None]],
                                     dim=-1)
            # gt_bboxes_3d = gt_bboxes_3d.tensor
            gt_bboxes_3d = gt_bboxes_3d.to(device)
        gt_bboxes_3d_ass_xyz = self.xyzrange2normxyz(gt_bboxes_3d[:, :3])
        gt_bboxes_3d_xz = torch.cat([gt_bboxes_3d_ass_xyz[:, :1], gt_bboxes_3d_ass_xyz[:, 2:3]], dim=-1)

        reg_out_ass_xyz = reg_out[:, :3]
        reg_out_xz = torch.cat([reg_out_ass_xyz[:, :1], reg_out_ass_xyz[:, 2:3]], dim=-1)


        assign_result = self.assigner.assign(reg_out_xz,cls_out, self.xyz_prior[0, :, [0, 2]],
                                              gt_bboxes_3d_xz,gt_labels_3d)

        sampling_result = self.sampler.sample(assign_result, reg_out, gt_bboxes_3d)
        single_img_pos_inds = sampling_result.pos_inds

        gt_bboxes_3d_target = gt_bboxes_3d#self.get_regtarget_from_gt(gt_bboxes_3d, gt_labels_3d)

        # label targets
        single_img_labels_target = gt_bboxes_3d.new_full((num_query,), self.num_classes, dtype=torch.long)
        single_img_labels_target[single_img_pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]

        # bbox targets
        single_img_bboxes3d_target = gt_bboxes_3d_target[sampling_result.pos_assigned_gt_inds]
        single_img_bboxes3d_target = single_img_bboxes3d_target.to(device)


        return single_img_labels_target, single_img_bboxes3d_target, single_img_pos_inds

    def get_regtarget_from_gt(self, gt_bboxes_3d, gt_labels_3d):
        gt_bboxes_3d_target_xyz = self.xyzrange2normxyz(gt_bboxes_3d[:, :3])  # gt_bboxes_3d[:, :3]
        gt_bboxes_3d_target_dim = torch.log(gt_bboxes_3d[:, 3:6] / self.canon_box_sizes[gt_labels_3d])
        gt_bboxes_3d_target_roty = gt_bboxes_3d[:, 6:]
        gt_bboxes_3d_target = torch.cat([gt_bboxes_3d_target_xyz, gt_bboxes_3d_target_dim, gt_bboxes_3d_target_roty],
                                        dim=-1)
        return gt_bboxes_3d_target

    def get_gt_from_reg(self, reg_bboxes_3d, reg_labels_3d,keep_dir=True):
        gt_bboxes_3d_target_xyz = self.normxyz2xyzrange(reg_bboxes_3d[:, :3])  # gt_bboxes_3d[:, :3]
        gt_bboxes_3d_target_dim = torch.exp(reg_bboxes_3d[:, 3:6]) * self.canon_box_sizes[reg_labels_3d]
        if keep_dir:
            gt_bboxes_3d_target_dir = reg_bboxes_3d[:,6:]
        else:
            gt_bboxes_3d_target_dir = torch.atan2(reg_bboxes_3d[:, 6:7], reg_bboxes_3d[:, 7:8])

        gt_bboxes_3d_target = torch.cat([gt_bboxes_3d_target_xyz, gt_bboxes_3d_target_dim, gt_bboxes_3d_target_dir],
                                        dim=-1)
        return gt_bboxes_3d_target

    @force_fp32(apply_to=('cls_out', 'reg_out'))
    def loss(self, cls_out, reg_out,
             gt_bboxes_3d,
             gt_labels_3d,
             img_metas, gt_bboxes_ignore=None, prefix=''):
        # cam2imgs = torch.stack([img_meta['cam2img'] for img_meta in img_metas], dim=0).to(cls_out.device)
        # reg_xyz_out = (reg_out[..., 0:3] + 1) * 0.5 * (self.xyzrange[:, 1] - self.xyzrange[:, 0])[
        #     None, None] + self.xyzrange[:, 0][None, None]
        decode_bboxes3d = torch.cat(
            [reg_out[..., 0:3], reg_out[..., 3:6], torch.atan2(reg_out[..., 6:7], reg_out[..., 7:8])],
            dim=-1)
        batch_target = self.get_targets(cls_out.detach(), reg_out.detach(), decode_bboxes3d.detach(),
                                        gt_bboxes_3d,
                                        gt_labels_3d,
                                        img_metas)
        loss_dict = self.get_loss_from_target(cls_out, reg_out, batch_target, prefix)

        return loss_dict

    def get_loss_from_target(self, cls_out, reg_out, batch_target, prefix='', dn=False):
        batch_labels_targets, batch_bboxes3d_targets, batch_pos_inds = batch_target
        batch_pos_binds = torch.cat(
            [pos_inds + cls_out.size(1) * batch_id for batch_id, pos_inds in enumerate(batch_pos_inds)])
        # batch_clspos_binds = torch.cat(
        #     [clspos_inds + cls_out.size(1) * batch_id for batch_id, clspos_inds in enumerate(batch_clspos_inds)])
        # batch_pos_bids = torch.cat(
        #     [torch.full_like(pos_inds, batch_id) for batch_id, pos_inds in enumerate(batch_pos_inds)])
        num_batch_pos = batch_pos_binds.size(0)
        batch_labels_targets = torch.cat(batch_labels_targets, dim=0)
        batch_bboxes3d_targets = torch.cat(batch_bboxes3d_targets, dim=0)

        batch_labels_weight = torch.ones_like(batch_labels_targets) * self.bg_cls_weight
        batch_labels_weight[batch_pos_binds] = 1.0
        # classification loss
        cls_out = cls_out.reshape(-1, self.num_classes + 1)
        reg_out = reg_out.reshape(-1, 8)
        loss_cls = self.loss_cls(
            cls_out, batch_labels_targets,
            weight=batch_labels_weight,
            avg_factor=max(num_batch_pos, 1)
        )
        mae_poscls = (cls_out.sigmoid()[batch_pos_binds][:,batch_labels_targets[batch_pos_binds]]).float().mean()
        mae_cls = (cls_out.max(-1)[1] == batch_labels_targets).float().mean()
        if num_batch_pos > 0:
            bboxes3d_pos = batch_bboxes3d_targets
            labels_pos = batch_labels_targets[batch_pos_binds]
            reg_pos = reg_out[batch_pos_binds]
            reg_pos = self.get_gt_from_reg(reg_pos,labels_pos,keep_dir=True)
            reg_xyz_pos = reg_pos[..., :3]
            reg_dim_pos = reg_pos[:, 3:6]
            reg_roty_pos = reg_pos[:, 6:]
            # reg_bboxes3d_xyz = torch.cat([reg_xyz_pos, bboxes3d_pos[..., 3:6], bboxes3d_pos[..., 6:]], dim=-1)
            # reg_bboxes3d_dim = torch.cat([bboxes3d_pos[:, :3], reg_dim_pos, bboxes3d_pos[:, 6:]], dim=-1)
            # decode_bboxes3d = torch.cat([reg_xyz_pos, reg_dim_pos, torch.atan2(reg_pos[..., 6:7], reg_pos[..., 7:8])],
            #                             dim=-1)

            loss_xyz = self.loss_reg(
                reg_xyz_pos,
                bboxes3d_pos[:, :3],
                avg_factor=max(num_batch_pos, 1)
            )/4
            loss_dim = self.loss_reg(
                reg_dim_pos,
                bboxes3d_pos[:, 3:6],
                avg_factor=max(num_batch_pos, 1)
            )
            loss_roty = self.loss_dir(
                reg_roty_pos,  # reg_pos[:, 6:],
                bboxes3d_pos[:, 6],
                avg_factor=max(num_batch_pos, 1)
            )
            loss_roty_norm = (torch.norm(reg_pos[:, 6:], dim=-1) - 1).abs().mean()  # L2 norm
            pred_bboxes = reg_pos
            gt_bboxes = bboxes3d_pos
            mae_xyz = (pred_bboxes[:, :3] - gt_bboxes[:, :3]).abs().mean()
            mae_dim = (pred_bboxes[:, 3:6] - gt_bboxes[:, 3:6]).abs().mean()
            mae_roty = loss_roty*10 / torch.pi * 180
            if not dn:
                print(cls_out[batch_pos_binds][:3].sigmoid().detach().cpu().numpy().tolist(),
                      labels_pos[:3])
                print(cls_out[:3].sigmoid().detach().cpu().numpy().tolist(),
                      batch_labels_targets[:3])
                print(pred_bboxes[0].detach().cpu().numpy().tolist())
                print(gt_bboxes[0].detach().cpu().numpy().tolist())
        else:
            loss_xyz = reg_out.sum() * 0
            loss_dim = loss_xyz
            loss_roty = loss_xyz
            loss_roty_norm = loss_xyz
            # loss_reg_xyzdim = loss_xyz
            mae_xyz = loss_xyz
            mae_dim = loss_xyz
            mae_roty = loss_xyz
        # loss_roty_norm = (torch.sum(reg_out[:, 6:] ** 2, dim=-1) - 1).abs().mean()

        print(mae_cls, mae_poscls, mae_xyz, mae_dim, mae_roty)
        losses = dict(
            loss_cls=loss_cls,
            loss_xyz=loss_xyz,
            loss_dim=loss_dim,
            loss_roty=loss_roty,
            loss_roty_norm=loss_roty_norm,

            mae_xyz=mae_xyz,
            mae_dim=mae_dim,
            mae_roty=mae_roty,
            mae_cls=mae_cls,
            mae_poscls=mae_poscls,
        )
        for k, v in losses.items():
            losses[prefix + k] = losses.pop(k)
        return losses

    def get_bev_priors(
            self, batch_size, bev_size, zrange, ux_div_fdx, dtype, device
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
        x_range = torch.arange(w, dtype=dtype, device=device) / w
        y = torch.ones((h, w, 1), dtype=dtype, device=device) * 0.67
        z_range = torch.arange(h, dtype=dtype, device=device) / h
        z, x = torch.meshgrid(z_range, x_range)
        # x = x * ux_div_fdx * z_range[:, None]
        x = x.view(h, w, 1)
        y = y.view(h, w, 1)
        z = z.view(h, w, 1)
        dim = torch.ones((h, w, 3), dtype=dtype, device=device)
        roty = torch.ones((h, w, 2), dtype=dtype, device=device)
        proiors = torch.cat([x, y, z, dim, roty], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1, 1)
