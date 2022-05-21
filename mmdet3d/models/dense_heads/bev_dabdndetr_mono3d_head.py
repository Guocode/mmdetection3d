# Copyright (c) OpenMMLab. All rights reserved.
import math
import random

import torch
from mmcv.cnn import ConvModule, normal_init, bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, build_assigner, build_sampler
from mmdet.models.utils import SinePositionalEncoding
from torch import nn
from torch.nn import functional as F

from mmdet3d.core import CameraInstance3DBoxes
from ..bev_transformer.conv_bev_transformer import ConvBEVTransformer
from mmdet.models.builder import HEADS, build_loss
from mmdet3d.models.dense_heads.base_mono3d_dense_head import BaseMono3DDenseHead


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
                 stacked_convs=3,
                 # loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_cls=dict(type='FocalLoss', loss_weight=1.0),
                 loss_dir=dict(type='DirCosineLoss', loss_weight=0.1),
                 loss_reg=dict(type='L1Loss'),
                 loss_iou3d=dict(type='DIOU3DLoss'),
                 bg_cls_weight=0.1,
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 group_reg_dims=(3, 3, 2),  # xyz:3,dim:3,rotysincos:2
                 xrange=(-40., 40.),
                 yrange=(-0.94, 3.00),
                 zrange=(0.3, 60.),
                 bev_prior_size=(64, 64),
                 num_query=100,
                 bev_transformer_chn=None,
                 bev_transformer_kernel_s=9,
                 bev_transformer_convs_num=3,
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

        self.yrange = yrange
        self.zrange = zrange
        self.register_buffer('xyzrange',
                             torch.asarray([xrange, yrange, zrange, self.xrange_min], dtype=torch.float32))  # (3,2)
        self.bev_prior_size = bev_prior_size
        self.num_query = num_query
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
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.dnassigner = build_assigner(self.train_cfg.dnassigner)

            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.bev_transformer = ConvBEVTransformer(in_channels=self.in_channels,
                                                  out_channels=self.feat_channels,
                                                  feat_channels=self.bev_transformer_chn,
                                                  stacked_bev_convs_num=self.bev_transformer_convs_num,
                                                  kernel_s=self.bev_transformer_kernel_s,
                                                  )
        self._init_clsreg_convs()
        positional_xyz_encoded = torch.stack(
            [*torch.meshgrid(torch.arange(0, 10) / 5 - 1, torch.arange(0, 10) / 5 - 1),
             torch.normal(0, 0.01, (10, 10))])
        self.positional_query = nn.Parameter(positional_xyz_encoded, requires_grad=True)  # (-1,1) related with xyz
        self.positional_query_embedding = nn.Sequential(
            nn.Conv2d(3, self.feat_channels, 1),
            nn.ReLU(),
            nn.Conv2d(self.feat_channels, self.feat_channels, 1),
            # nn.Sigmoid()
        )

        self.content_query = nn.Parameter(torch.normal(0, 0.001, (1, self.cls_out_channels + 3 + 2, 10, 10)),
                                          requires_grad=True)  # related with cls+dxdydz+roty(sincos)
        self.content_query_embedding = nn.Sequential(
            nn.Conv2d(self.cls_out_channels + 3 + 2, self.feat_channels, 1),
            nn.ReLU(),
            nn.Conv2d(self.feat_channels, self.feat_channels, 1),
            nn.Sigmoid()
        )

        self.bev_enc = nn.Sequential(
            ConvModule(
                self.feat_channels,
                self.feat_channels,
                3,
                padding=1
            ),
            nn.Dropout(p=0.1),
            ConvModule(
                self.feat_channels,
                self.feat_channels,
                3,
                padding=1
            ),
        )
        self.ffn = nn.Conv2d(self.feat_channels, self.cls_out_channels + self.reg_out_channels, 1)

        self.init_weights()

    def make_pseudo_positional_query(self, gt_bboxes3d, gt_labels, xyz_delta=2.):
        '''

        :param gt_bboxes3d: List of N*7, N is number of bboxes3d
        :param gt_labels:  List of N, N is number of bboxes3d
        :return: pseudo_positional_query: B*num_query*3
        '''
        bs = len(gt_bboxes3d)
        pseudo_positional_query = self.positional_query.repeat(bs, 1, 1, 1).detach()
        for i in range(bs):
            for bid, gt_bbox3d in enumerate(gt_bboxes3d):
                gt_xyz = gt_bbox3d[:3]
                pseudo_positional_query[i, bid // 10, bid % 10,:3] = gt_xyz
        return pseudo_positional_query
    def make_pseudo_content_query(self, gt_bboxes3d, gt_labels, label_flip=0.2, dim_delta=0.5, roty_delta=0.5):
        bs = len(gt_bboxes3d)
        pseudo_content_query = self.content_query.repeat(bs, 1, 1, 1).detach()
        for i in range(bs):
            query_ids = list(range(self.num_query))
            querys_selected = random.sample(query_ids,k=len(gt_bboxes3d[i]))
            for bid, gt_bbox3d in enumerate(gt_bboxes3d):
                # gt_xyz = gt_bbox3d[:3]
                gt_dim = gt_bbox3d[3:6]
                gt_roty = gt_bbox3d[6:7]
                if random.random()<label_flip:
                    labels = list(range(self.num_classes+1))
                    labels.pop(gt_labels[i][bid].item())
                    gt_label = random.choice(labels)
                else:
                    gt_label = gt_labels[i][bid]

                pseudo_content_query[i, bid // 10, bid % 10,gt_label] = 1
                pseudo_content_query[i, bid // 10, bid % 10,3:6] = gt_dim
                pseudo_content_query[i, bid // 10, bid % 10,6:8] = gt_roty.sin(),gt_roty.cos()

        return pseudo_content_query
    def get_pseudo_query(self,gt_bboxes3d,gt_labels):
        gt_bboxes3d = random.shuffle(gt_bboxes3d)

    def _init_clsreg_convs(self):
        """Initialize classification conv layers of the head."""
        clsreg_convs = nn.ModuleList()
        chout = self.feat_channels
        for i in range(self.stacked_convs):
            chin = chout
            chout = self.feat_channels
            conv_cfg = self.conv_cfg
            clsreg_convs.append(
                ConvModule(
                    chin,
                    chout,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias),
            )
            clsreg_convs.append(
                CBAMBlock("FC", 3, channels=chout, ratio=chout // 4)
            )
        self.clsreg_convs = nn.Sequential(*clsreg_convs)

    def init_weights(self):
        """Initialize weights of the head.

        We currently still use the customized defined init_weights because the
        default init of DCN triggered by the init_cfg will init
        conv_offset.weight, which mistakenly affects the training stability.
        """
        for bev_enc in self.bev_enc:
            normal_init(bev_enc, std=0.00001, bias=0.00001)
        for conv_cls in self.clsreg_convs:
            normal_init(conv_cls, std=0.01)
        normal_init(self.ffn, std=0.0001, bias=0.0001)
        # for conv_reg in self.reg_convs:
        #     normal_init(conv_reg, std=0.01)

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
        # self.register_buffer('canon_box_sizes', )
        self.canon_box_sizes = nn.Parameter(canon)

    def sigmoidxyz2xyzrange(self, sigmoidxyz):
        # xyz: B,N,3
        shape = sigmoidxyz.shape
        sigmoidxyz = sigmoidxyz.reshape(-1, 3)
        yr = sigmoidxyz[:, 1:2] * (self.xyzrange[1:2, 1] - self.xyzrange[1:2, 0])[None] + self.xyzrange[1:2, 0][None]
        zr = sigmoidxyz[:, 2:3] * (self.xyzrange[2:3, 1] - self.xyzrange[2:3, 0])[None] + self.xyzrange[2:3, 0][None]
        cxr = self.xyzrange[3][None] + (self.xyzrange[0] - self.xyzrange[3])[None] * sigmoidxyz[:, 2:3]
        xr = sigmoidxyz[:, 0:1] * (cxr[:, 1:2] - cxr[:, 0:1]) + cxr[:, 0:1]
        xyzr = torch.cat([xr, yr, zr], dim=-1)
        xyzr = xyzr.reshape(*shape)
        return xyzr  # xyz * (self.xyzrange[:3, 1] - self.xyzrange[:3, 0])[None, None] + self.xyzrange[:3, 0][None, None]

    def xyzrange2sigmoidxyz(self, xyzrange):
        # xyz: N,3
        shape = xyzrange.shape
        xyzrange = xyzrange.reshape(-1, 3)
        ys = (xyzrange[:, 1:2] - self.xyzrange[1:2, 0][None]) / (self.xyzrange[1:2, 1] - self.xyzrange[1:2, 0])[None]
        zs = (xyzrange[:, 2:3] - self.xyzrange[2:3, 0][None]) / (self.xyzrange[2:3, 1] - self.xyzrange[2:3, 0])[None]
        cxr = self.xyzrange[3][None] + (self.xyzrange[0] - self.xyzrange[3])[None] * zs
        xs = (xyzrange[:, 0:1] - cxr[:, 0:1]) / (cxr[:, 1:2] - cxr[:, 0:1])
        xyzs = torch.cat([xs, ys, zs], dim=-1)
        xyzs = xyzs.reshape(*shape)
        return xyzs  # (xyzrange - self.xyzrange[:, 0][None]) / (self.xyzrange[:, 1] - self.xyzrange[:, 0])[None]

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
        loss_inputs = outs + (gt_bboxes_3d,
                              gt_labels,
                              img_metas)
        losses = self.loss(*loss_inputs)

        if proposal_cfg is None:
            return losses
        else:
            raise NotImplementedError

    def forward(self, feats, content_query=None, positional_query=None):
        feats_singlelvl = feats[0]  #
        bev_feats = self.bev_transformer(feats_singlelvl)
        bev_feats = F.adaptive_avg_pool2d(bev_feats, (10, 10))
        b, c, h, w = bev_feats.shape
        if content_query is None:
            content_query = self.content_query
        if positional_query is None:
            positional_query = self.positional_query
        bev_query = bev_feats * (self.content_query_embedding(content_query) * 2)
        postional_query_embed = self.positional_query_embedding(positional_query)
        # bev_query = torch.sigmoid(bev_feats)
        for clsreg_conv in self.clsreg_convs:
            bev_query = postional_query_embed + bev_query + self.bev_enc(clsreg_conv(bev_query))
        bev_query = self.ffn(bev_query)
        cls_reg_out = bev_query.view(b, -1, self.num_query).permute(0, 2, 1)
        cls_out, reg_out = cls_reg_out[..., :self.cls_out_channels], cls_reg_out[..., self.cls_out_channels:]
        b, nq, c = cls_out.shape
        cls_out = cls_out.view(b, self.num_query, self.num_classes + 1)
        b, nq, c = reg_out.shape
        reg_out = reg_out.view(b, self.num_query, sum(self.group_reg_dims))
        reg_xyz_out = self.sigmoidxyz2xyzrange(torch.sigmoid(reg_out[..., 0:3]))  # (-1,1)
        reg_dim_out = (torch.sigmoid(reg_out[..., 3:6]) * 2).clamp(0.3, 2)  # (0,2)
        reg_roty_out = F.normalize(torch.sigmoid(reg_out[..., 6:8] * 2 - 1), dim=-1)
        regs_out = torch.cat([reg_xyz_out, reg_dim_out, reg_roty_out], dim=-1)
        return cls_out, regs_out

    @force_fp32(apply_to=('cls_out', 'reg_out'))
    def get_bboxes(self, cls_out, reg_out, img_metas, rescale=None):
        result_list = []
        for img_id in range(len(img_metas)):
            scores, labels = torch.softmax(cls_out[img_id], dim=-1).max(-1)
            pos_idx = (scores > 0.25) * (labels < self.num_classes)
            scores = scores[pos_idx]
            labels = labels[pos_idx]

            bboxes3d = reg_out[img_id][pos_idx]
            bboxes3d_dim = bboxes3d[:, 3:6] * self.canon_box_sizes[labels]
            bboxes3d = torch.cat([bboxes3d[:, :3], bboxes3d_dim, torch.atan2(bboxes3d[:, 6:7], bboxes3d[:, 7:8])],
                                 dim=-1)
            bboxes3d = CameraInstance3DBoxes(bboxes3d, box_dim=7, origin=(0.5, 1.0, 0.5))
            bboxes2d = torch.cat(
                [torch.zeros((scores.size(0), 4), dtype=scores.dtype, device=scores.device), scores.view(-1, 1)],
                dim=-1)
            result_list.append((bboxes2d, bboxes3d, labels))
        return result_list

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
        # batch_dntargets_res = multi_apply(
        #     self.get_dntargets_single_img,
        #     cls_out,
        #     reg_out,
        #     decode_bboxes3d,
        #     gt_bboxes_3d,
        #     gt_labels_3d,
        #     img_metas
        # )
        return batch_targets_res  # + batch_dntargets_res

    @torch.no_grad()
    def get_targets_single_img(self, cls_out, reg_out, decode_bboxes3d, gt_bboxes_3d, gt_labels_3d, img_meta):
        # assigner and sampler
        device = cls_out.device
        if isinstance(gt_bboxes_3d, CameraInstance3DBoxes):  # target bbox3d with gravity center and alpha
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(device)
        gt_bboxes_3d_ass_xyz = self.xyzrange2sigmoidxyz(gt_bboxes_3d[:, :3]).clamp(0, 1)
        gt_bboxes_3d_ass_xyz = torch.cat(
            [gt_bboxes_3d_ass_xyz[:, :1], gt_bboxes_3d_ass_xyz[:, 1:2] - gt_bboxes_3d_ass_xyz[:, 1:2],
             gt_bboxes_3d_ass_xyz[:, 2:3]], dim=-1)
        gt_bboxes_3d_ass_dim = gt_bboxes_3d[:, 3:6] / gt_bboxes_3d[:, 3:6] / 60
        gt_bboxes_3d_ass_roty = gt_bboxes_3d[:, 6:] - gt_bboxes_3d[:, 6:]
        gt_bboxes_3d_ass = torch.cat([gt_bboxes_3d_ass_xyz, gt_bboxes_3d_ass_dim, gt_bboxes_3d_ass_roty], dim=-1)

        decode_ass_xyz = self.xyzrange2sigmoidxyz(decode_bboxes3d[:, :3]).clamp(0, 1)
        decode_ass_xyz = torch.cat(
            [decode_ass_xyz[:, :1], decode_ass_xyz[:, 1:2] - decode_ass_xyz[:, 1:2], decode_ass_xyz[:, 2:3]], dim=-1)
        decode_ass_dim = decode_bboxes3d[:, 3:6] / decode_bboxes3d[:, 3:6] / 60  # / 30  # / decode_bboxes3d[:, 3:6]
        decode_ass_roty = decode_bboxes3d[:, 6:] - decode_bboxes3d[:, 6:]
        decode_bboxes3d_ass = torch.cat([decode_ass_xyz, decode_ass_dim, decode_ass_roty], dim=-1)

        gt_bboxes_3d_target_xyz = self.sigmoidxyz2xyzrange(gt_bboxes_3d_ass_xyz)  # gt_bboxes_3d[:, :3]
        gt_bboxes_3d_target_dim = gt_bboxes_3d[:, 3:6]
        gt_bboxes_3d_target_roty = gt_bboxes_3d[:, 6:]
        gt_bboxes_3d_target = torch.cat([gt_bboxes_3d_target_xyz, gt_bboxes_3d_target_dim, gt_bboxes_3d_target_roty],
                                        dim=-1)
        dnassign_result = self.dnassigner.assign(decode_bboxes3d_ass, cls_out, gt_bboxes_3d_ass,
                                                 gt_labels_3d, img_meta, )
        dnsampling_result = self.sampler.sample(dnassign_result, reg_out, gt_bboxes_3d)
        dnsingle_img_pos_inds = dnsampling_result.pos_inds

        single_img_labels_dntarget = gt_bboxes_3d.new_full((self.num_query,), self.num_classes, dtype=torch.long)
        single_img_labels_dntarget[dnsingle_img_pos_inds] = gt_labels_3d[dnsampling_result.pos_assigned_gt_inds]

        single_img_bboxes3d_dntarget = torch.cat([reg_out[:, :6], torch.atan2(reg_out[:, 6:7], reg_out[:, 7:8])],
                                                 dim=-1)
        single_img_bboxes3d_dntarget[dnsingle_img_pos_inds] = gt_bboxes_3d_target[
            dnsampling_result.pos_assigned_gt_inds]
        if dnassign_result.max_overlaps is not None:
            single_img_bboxes3d_dnweight = dnassign_result.max_overlaps.clamp(0., 1.)
        else:
            single_img_bboxes3d_dnweight = torch.zeros(reg_out.size(0), dtype=torch.float32, device=reg_out.device)

        assign_result = self.assigner.assign(decode_bboxes3d_ass, cls_out, gt_bboxes_3d_ass,
                                             gt_labels_3d, img_meta, )
        sampling_result = self.sampler.sample(assign_result, reg_out, gt_bboxes_3d)
        single_img_pos_inds = sampling_result.pos_inds
        # print('single_img_pos_inds', single_img_pos_inds)
        # label targets
        single_img_labels_target = gt_bboxes_3d.new_full((self.num_query,), self.num_classes, dtype=torch.long)
        single_img_labels_target[single_img_pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]

        # bbox targets
        single_img_bboxes3d_target = gt_bboxes_3d_target[sampling_result.pos_assigned_gt_inds]
        single_img_bboxes3d_target = single_img_bboxes3d_target.to(device)

        # hungarian prior
        single_img_labels_dntarget[single_img_pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
        single_img_bboxes3d_dntarget[single_img_pos_inds] = gt_bboxes_3d_target[sampling_result.pos_assigned_gt_inds]
        return single_img_labels_target, single_img_bboxes3d_target, single_img_pos_inds, \
               single_img_labels_dntarget, single_img_bboxes3d_dntarget, dnsingle_img_pos_inds, single_img_bboxes3d_dnweight

    @force_fp32(apply_to=('cls_out', 'reg_out'))
    def loss(self, cls_out, reg_out,
             gt_bboxes_3d,
             gt_labels_3d,
             img_metas, gt_bboxes_ignore=None):

        decode_bboxes3d = torch.cat(
            [reg_out[..., 0:3], reg_out[..., 3:6], torch.atan2(reg_out[..., 6:7], reg_out[..., 7:8])],
            dim=-1)
        batch_target = self.get_targets(cls_out.detach(), reg_out.detach(), decode_bboxes3d.detach(),
                                        gt_bboxes_3d,
                                        gt_labels_3d,
                                        img_metas)
        loss_dict = self.get_loss_from_target(cls_out, reg_out, batch_target)

        return loss_dict

    def get_loss_from_target(self, cls_out, reg_out, batch_target):
        batch_labels_targets, batch_bboxes3d_targets, batch_pos_inds, batch_labels_dntargets, batch_bboxes3d_dntargets, batch_pos_dninds, dnweight = batch_target
        batch_pos_binds = torch.cat(
            [pos_inds + self.num_query * batch_id for batch_id, pos_inds in enumerate(batch_pos_inds)])
        batch_pos_dnbinds = torch.cat(
            [pos_inds + self.num_query * batch_id for batch_id, pos_inds in enumerate(batch_pos_dninds)])
        # batch_pos_bids = torch.cat(
        #     [torch.full_like(pos_inds, batch_id) for batch_id, pos_inds in enumerate(batch_pos_inds)])
        num_batch_pos = batch_pos_binds.size(0)
        num_batch_dnpos = batch_pos_dnbinds.size(0)
        batch_labels_targets = torch.cat(batch_labels_targets, dim=0)
        batch_bboxes3d_targets = torch.cat(batch_bboxes3d_targets, dim=0)
        batch_labels_dntargets = torch.cat(batch_labels_dntargets, dim=0)
        batch_bboxes3d_dntargets = torch.cat(batch_bboxes3d_dntargets, dim=0)
        dnweight = torch.cat(dnweight, dim=0)

        batch_labels_weight = torch.ones_like(batch_labels_targets) * self.bg_cls_weight
        batch_labels_weight[batch_pos_binds] = 1.0
        # classification loss
        cls_out = cls_out.reshape(-1, self.num_classes + 1)
        reg_out = reg_out.reshape(-1, 8)
        loss_cls = self.loss_cls(
            cls_out, batch_labels_targets,
            weight=batch_labels_weight,
            avg_factor=num_batch_pos * 1.0 + (cls_out.size(0) - num_batch_pos) * self.bg_cls_weight
        )
        if num_batch_pos > 0:
            bboxes3d_pos = batch_bboxes3d_targets
            labels_pos = batch_labels_targets[batch_pos_binds]
            reg_pos = reg_out[batch_pos_binds]
            reg_xyz_pos = reg_pos[..., :3]
            # reg_dim_pos = reg_pos[:, 3:6]
            reg_dim_pos = (reg_pos[:, 3:6]) * self.canon_box_sizes[labels_pos]
            decode_bboxes3d_xyz = torch.cat([reg_xyz_pos, bboxes3d_pos[..., 3:]], dim=-1)
            decode_bboxes3d_dim = torch.cat([bboxes3d_pos[:, :3], reg_dim_pos, bboxes3d_pos[:, 6:]], dim=-1)
            decode_bboxes3d = torch.cat([reg_xyz_pos, reg_dim_pos, torch.atan2(reg_pos[..., 6:7], reg_pos[..., 7:8])],
                                        dim=-1)

            # reg_xyz_target = (batch_bboxes3d_targets[:, :3] - self.xyzrange[:, 0][None]) / (
            #         self.xyzrange[:, 1] - self.xyzrange[:, 0]) * 2 - 1
            # batch_bboxes3d_targets = torch.cat([reg_xyz_target, batch_bboxes3d_targets[:, 3:]], dim=-1)
            # print(batch_pos_binds)
            # print(cls_out[:,:3])
            print(F.softmax(cls_out[batch_pos_binds][:3],
                            dim=-1).detach().cpu().numpy().tolist(),
                  labels_pos[:3])
            print(decode_bboxes3d[0].detach().cpu().numpy().tolist())
            print(batch_bboxes3d_targets[0].detach().cpu().numpy().tolist())
            # print(self.bev_query[0, 4])
            loss_xyz = self.loss_iou3d(
                decode_bboxes3d_xyz,
                bboxes3d_pos,
                avg_factor=max(num_batch_pos, 1)
            )
            # loss_xy = self.loss_reg(
            #     decode_bboxes3d_xyz[:, :2]/bboxes3d_pos[:,2:3],
            #     bboxes3d_pos[:, :2]/bboxes3d_pos[:,2:3],
            #     avg_factor=max(num_batch_pos, 1)
            # )
            # loss_z =self.loss_reg(
            #     decode_bboxes3d_xyz[:,2:3],
            #     bboxes3d_pos[:,2:3],
            #     avg_factor=max(num_batch_pos, 1)
            # )

            loss_dim = self.loss_iou3d(
                decode_bboxes3d_dim,
                bboxes3d_pos,
                avg_factor=max(num_batch_pos, 1)
            )
            # loss_reg_xyzdim = self.loss_reg(
            #     reg_pos[:, :6],
            #     batch_bboxes3d_targets[:, :6],
            #     avg_factor=max(num_batch_pos, 1)
            # )
            loss_roty = self.loss_dir(
                reg_pos[:, 6:],
                batch_bboxes3d_targets[:, 6],
                avg_factor=max(num_batch_pos, 1)
            )
            mae_xyz = (decode_bboxes3d[:, :3] - batch_bboxes3d_targets[:, :3]).abs().mean()
            mae_dim = (decode_bboxes3d[:, 3:6] - batch_bboxes3d_targets[:, 3:6]).abs().mean()
            mae_roty = loss_roty * 10 / torch.pi * 180

        else:
            loss_xyz = reg_out.sum() * 0
            # loss_xy = loss_xyz
            # loss_z = loss_xyz

            loss_dim = loss_xyz
            loss_roty = loss_xyz
            loss_reg_xyzdim = loss_xyz
            mae_xyz = loss_xyz
            mae_dim = loss_xyz
            mae_roty = loss_xyz

        reg_dnout = reg_out
        dn_reg_weight = dnweight  # * (1 - F.softmax(cls_out, dim=-1).max(-1)[0])
        if num_batch_dnpos > 0:
            dn_reg_weight = dn_reg_weight[batch_pos_dnbinds]
            bboxes3d_dnpos = batch_bboxes3d_dntargets[batch_pos_dnbinds]
            labels_dnpos = batch_labels_dntargets[batch_pos_dnbinds]
            reg_dnpos = reg_dnout[batch_pos_dnbinds]
            reg_xyz_dnpos = reg_dnpos[..., :3]
            reg_dim_dnpos = (reg_dnpos[:, 3:6]) * self.canon_box_sizes[labels_dnpos]
            decode_bboxes3d_dnxyz = torch.cat([reg_xyz_dnpos, bboxes3d_dnpos[..., 3:]], dim=-1)
            decode_bboxes3d_dndim = torch.cat([bboxes3d_dnpos[:, :3], reg_dim_dnpos, bboxes3d_dnpos[:, 6:]], dim=-1)
            # decode_bboxes3d = torch.cat(
            #     [reg_xyz_dnpos, reg_dim_dnpos, torch.atan2(reg_dnpos[..., 6:7], reg_dnpos[..., 7:8])],
            #     dim=-1)

            loss_dnxyz = self.loss_iou3d(
                decode_bboxes3d_dnxyz,
                bboxes3d_dnpos,
                weight=dn_reg_weight,
                avg_factor=max(num_batch_dnpos, 1)
            )
            loss_dndim = self.loss_iou3d(
                decode_bboxes3d_dndim,
                bboxes3d_dnpos,
                weight=dn_reg_weight,
                avg_factor=max(num_batch_dnpos, 1)
            )
            loss_dnroty = self.loss_dir(
                reg_dnpos[:, 6:],
                bboxes3d_dnpos[:, 6],
                weight=dn_reg_weight,
                avg_factor=max(num_batch_dnpos, 1)
            )
        else:
            loss_dnxyz = reg_out.sum() * 0
            loss_dndim = loss_dnxyz
            loss_dnroty = loss_dnxyz

        print(mae_xyz, mae_dim, mae_roty)
        return dict(
            loss_cls=loss_cls,
            loss_xyz=loss_xyz,
            # loss_xy=loss_xy,
            # loss_z=loss_z,

            loss_dim=loss_dim,
            loss_roty=loss_roty,
            loss_xyzdn=loss_dnxyz,
            loss_dimdn=loss_dndim,
            # loss_regdn_xyzdim=loss_regdn_xyzdim,
            loss_rotydn=loss_dnroty,
            mae_xyz=mae_xyz,
            mae_dim=mae_dim,
            mae_roty=mae_roty,

        )

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
