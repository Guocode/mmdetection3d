# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
from mmcv.cnn import ConvModule, normal_init, bias_init_with_prob
from mmdet.core import multi_apply, build_assigner, build_sampler
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
class BEVDETRMono3DHead(BaseMono3DDenseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=3,
                 loss_cls=dict(type='FocalLoss', loss_weight=1.0),
                 loss_dir=dict(type='DirCosineLoss'),
                 loss_iou3d=dict(type='DIOU3DLoss'),
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 group_reg_dims=(3, 3, 2),  # xyz:3,dim:3,rotysincos:2
                 xrange=(-40., 40.),
                 yrange=(-0.94, 3.00),
                 zrange=(0.3, 100.),
                 num_query=100,
                 bev_transformer_chn=128,
                 bev_transformer_kernel_s=9,
                 bev_transformer_convs_num=3,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(BEVDETRMono3DHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = (num_classes + 1) * num_query
        self.reg_out_channels = sum(group_reg_dims) * num_query
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.group_reg_dims = group_reg_dims
        self.conv_bias = conv_bias
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange
        self.register_buffer('xyzrange', torch.asarray([xrange, yrange, zrange], dtype=torch.float32))
        self.num_query = num_query
        self.bev_transformer_chn = bev_transformer_chn
        self.bev_transformer_convs_num = bev_transformer_convs_num
        self.bev_transformer_kernel_s = bev_transformer_kernel_s
        self.loss_cls = build_loss(loss_cls)
        self.loss_dir = build_loss(loss_dir)
        self.loss_iou3d = build_loss(loss_iou3d)
        self._init_layers()
        self._init_canon_box_sizes()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.bev_transformer = ConvBEVTransformer(in_channels=self.in_channels,
                                                  out_channels=self.in_channels,
                                                  feat_channels=self.bev_transformer_chn,
                                                  stacked_bev_convs_num=self.bev_transformer_convs_num,
                                                  kernel_s=self.bev_transformer_kernel_s,
                                                  )
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()
        self.priors = False

        self.bev_enc = nn.Conv2d(2, self.in_channels, 1)

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        cls_convs = nn.ModuleList()
        chout = self.in_channels
        for i in range(self.stacked_convs):
            chin = chout
            chout = chin * 2
            conv_cfg = self.conv_cfg
            cls_convs.append(
                ConvModule(
                    chin,
                    chout,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias),
            )
            cls_convs.append(
                CBAMBlock("FC", 5, channels=chout, ratio=chout // 4)
            )
        self.cls_convs = nn.Sequential(*cls_convs)

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        reg_convs = nn.ModuleList()
        chout = self.in_channels
        for i in range(self.stacked_convs):
            chin = chout
            chout = chin * 2
            conv_cfg = self.conv_cfg
            reg_convs.append(
                ConvModule(
                    chin,
                    chout,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias)
            )
            reg_convs.append(
                CBAMBlock("FC", 5, channels=chout, ratio=chout // 4)
            )
        self.reg_convs = nn.Sequential(*reg_convs)

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Linear(self.in_channels * (2 ** self.stacked_convs),
                                  self.cls_out_channels + self.reg_out_channels)
        # self.conv_regs = nn.Linear(self.in_channels * (2 ** self.stacked_convs), self.reg_out_channels)

    def init_weights(self):
        """Initialize weights of the head.

        We currently still use the customized defined init_weights because the
        default init of DCN triggered by the init_cfg will init
        conv_offset.weight, which mistakenly affects the training stability.
        """

        normal_init(self.conv_cls, std=0.01, bias=bias_init_with_prob(0.01))
        # normal_init(self.conv_regs, std=0.01, bias=bias_init_with_prob(0.01))
        for conv_cls in self.cls_convs:
            normal_init(conv_cls, std=0.01)
        for conv_reg in self.reg_convs:
            normal_init(conv_reg, std=0.01)

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
        if not self.priors:
            bev_ltcenterxz_priors = self.get_bev_ltcenterxz_priors(1, x[0].shape[2:], self.zrange,
                                                                   torch.stack(
                                                                       [img_meta['cam2img'] for img_meta in
                                                                        img_metas]).to(x[0].device),
                                                                   torch.float32, x[0].device, flatten=False)
            self.register_buffer('bev_ltcenterxz_priors', bev_ltcenterxz_priors.permute(0, 3, 1, 2))
            self.priors = True
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
        feats_singlelvl = feats[0]
        bev_feats = self.bev_transformer(feats_singlelvl)
        b, c, h, w = bev_feats.shape
        bev_feats = bev_feats + self.bev_enc(self.bev_ltcenterxz_priors)
        # bev_feats = self.bev_pool(bev_feats)
        cls_reg_feat = bev_feats
        # reg_feat = bev_feats
        cls_reg_feat = self.cls_convs(cls_reg_feat)
        cls_reg_feat = F.adaptive_avg_pool2d(cls_reg_feat, 1)

        cls_reg_out = self.conv_cls(cls_reg_feat.view(b, -1))
        cls_out, reg_out = cls_reg_out[:, :self.cls_out_channels], cls_reg_out[:, self.cls_out_channels:]
        # for cls_layer in self.cls_convs:
        #     cls_feat = cls_layer(cls_feat)
        # cls_feat = F.adaptive_avg_pool2d(cls_feat, 1)
        # cls_out = self.conv_cls(cls_feat.view(b, -1))
        b, c = cls_out.shape
        cls_out = cls_out.view(b, self.num_query, self.num_classes + 1)

        # for reg_layer in self.reg_convs:
        #     reg_feat = reg_layer(reg_feat)
        # reg_feat = F.adaptive_avg_pool2d(reg_feat, 1)
        # regs_out = self.conv_regs(reg_feat.view(b, -1))
        b, c = reg_out.shape
        reg_out = reg_out.view(b, self.num_query, sum(self.group_reg_dims))
        reg_xyz_out = (torch.tanh(reg_out[..., 0:3]) + 1) * 0.5 * (self.xyzrange[:, 1] - self.xyzrange[:, 0])[
            None, None] + self.xyzrange[:, 0][None, None]
        reg_dim_out = reg_out[..., 3:6]
        reg_roty_out = F.normalize(reg_out[..., 6:8], dim=-1)
        regs_out = torch.cat([reg_xyz_out, reg_dim_out, reg_roty_out], dim=-1)
        return cls_out, regs_out

    def _init_canon_box_sizes(self):
        self.register_buffer('canon_box_sizes', torch.asarray(
            [[0.62806586, 0.82038497, 1.76784787],  # Pedestrian
             [0.56898187, 1.77149234, 1.7237099],  # Cyclist
             [1.61876949, 3.89154523, 1.52969237],  # Car
             [1.9134491, 5.15499603, 2.18998422],  # Van
             [2.61168401, 9.22692319, 3.36492722],  # Truck
             [0.5390196, 1.08098042, 1.28392158],  # Person_sitting
             [2.36044838, 15.56991038, 3.5289238],  # Tram
             [1.24489164, 2.51495357, 1.61402478],  # Misc
             ])[:, [1, 2, 0]])

    def get_bboxes(self, cls_out, reg_out, img_metas, rescale=None):
        result_list = []
        for img_id in range(len(img_metas)):
            scores, labels = torch.softmax(cls_out[img_id], dim=-1).max(-1)
            pos_idx = (scores > 0.25) * (labels < self.num_classes)
            scores = scores[pos_idx]
            labels = labels[pos_idx]

            bboxes3d = reg_out[img_id][pos_idx]
            bboxes3d[:, 3:6] = (torch.tanh(bboxes3d[:, 3:6]) + 1) * self.canon_box_sizes[labels]
            bboxes3d = torch.cat([bboxes3d[:, :6], torch.atan2(bboxes3d[:, 6:7], bboxes3d[:, 7:8])], dim=-1)
            bboxes3d = CameraInstance3DBoxes(bboxes3d, box_dim=7, origin=(0.5, 1.0, 0.5))
            attrs = None
            result_list.append((bboxes3d, scores, labels, attrs))
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
        return batch_targets_res

    @torch.no_grad()
    def get_targets_single_img(self, cls_out, reg_out, decode_bboxes3d, gt_bboxes_3d, gt_labels_3d, img_meta):
        # assigner and sampler
        device = cls_out.device
        if isinstance(gt_bboxes_3d, CameraInstance3DBoxes):  # target bbox3d with gravity center and alpha
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(device)
        assign_result = self.assigner.assign(decode_bboxes3d, cls_out, gt_bboxes_3d,
                                             gt_labels_3d, img_meta, )
        sampling_result = self.sampler.sample(assign_result, reg_out, gt_bboxes_3d)
        single_img_pos_inds = sampling_result.pos_inds
        # label targets
        single_img_labels_target = gt_bboxes_3d.new_full((self.num_query,), self.num_classes, dtype=torch.long)
        single_img_labels_target[single_img_pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]

        # bbox targets
        single_img_bboxes3d_target = gt_bboxes_3d[sampling_result.pos_assigned_gt_inds]
        single_img_bboxes3d_target = single_img_bboxes3d_target.to(device)
        return single_img_labels_target, single_img_bboxes3d_target, single_img_pos_inds

    def loss(self, cls_out, reg_out,
             gt_bboxes_3d,
             gt_labels_3d,
             img_metas, gt_bboxes_ignore=None):
        cam2imgs = torch.stack([img_meta['cam2img'] for img_meta in img_metas], dim=0).to(cls_out.device)
        decode_bboxes3d = torch.cat([reg_out[..., :6], torch.atan2(reg_out[..., 6:7], reg_out[..., 7:8])], dim=-1)
        batch_target = self.get_targets(cls_out.detach(), reg_out.detach(), decode_bboxes3d.detach(),
                                        gt_bboxes_3d,
                                        gt_labels_3d,
                                        img_metas)
        loss_dict = self.get_loss_from_target(cls_out, reg_out, batch_target)

        return loss_dict

    def get_bev_ltcenterxz_priors(
            self, batch_size, bev_size, zrange, cam2imgs, dtype, device, flatten=True,
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
        ux_div_fdx = cam2imgs[0, 0:1, 2:3] / cam2imgs[0, 0:1, 0:1]
        h, w = bev_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) / w * 2 - 1
        z_range = (1 - (torch.arange(h, dtype=dtype, device=device)) / h) * (zrange[1] - zrange[0]) + zrange[0]
        z, x = torch.meshgrid(z_range, x_range)
        x = x * ux_div_fdx * z_range[None, :, None]
        x = x.view(z.shape)
        if flatten:
            z = z.flatten()
            x = x.flatten()
        # strides = x.new_full((x.shape[0],), stride)
        proiors = torch.stack([x, z], dim=-1)
        if flatten:
            return proiors.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            return proiors.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def get_loss_from_target(self, cls_out, reg_out, batch_target):

        batch_labels_target, batch_bboxes3d_targets, batch_pos_inds = batch_target
        batch_pos_binds = torch.cat(
            [pos_inds + self.num_query * batch_id for batch_id, pos_inds in enumerate(batch_pos_inds)])
        # batch_pos_bids = torch.cat(
        #     [torch.full_like(pos_inds, batch_id) for batch_id, pos_inds in enumerate(batch_pos_inds)])
        num_batch_pos = batch_pos_binds.size(0)

        batch_labels_target = torch.cat(batch_labels_target, dim=0)
        batch_bboxes3d_targets = torch.cat(batch_bboxes3d_targets, dim=0).detach()

        # classification loss
        cls_out = cls_out.reshape(-1, self.num_classes + 1)
        reg_out = reg_out.reshape(-1, 8)
        loss_cls = self.loss_cls(cls_out, batch_labels_target, avg_factor=max(num_batch_pos, 1))
        if num_batch_pos > 0:
            reg_pos = reg_out[batch_pos_binds]
            reg_xyz_pos = reg_pos[..., :3]
            red_dim_pos = (torch.tanh(reg_pos[:, 3:6]) + 1) * self.canon_box_sizes[
                batch_labels_target[batch_pos_binds]]
            decode_bboxes3d_xyz = torch.cat([reg_xyz_pos, batch_bboxes3d_targets[..., 3:]], dim=-1)
            decode_bboxes3d_dim = torch.cat(
                [batch_bboxes3d_targets[..., :3], red_dim_pos, batch_bboxes3d_targets[..., 6:]], dim=-1)
            decode_bboxes3d = torch.cat([reg_xyz_pos, red_dim_pos, torch.atan2(reg_pos[..., 6:7], reg_pos[..., 7:8])],
                                        dim=-1)
            print(cls_out.view(-1, self.num_classes + 1)[batch_pos_binds][0].detach().cpu().numpy().tolist(),
                  num_batch_pos)
            print(decode_bboxes3d[0].detach().cpu().numpy().tolist())
            print(batch_bboxes3d_targets[0].detach().cpu().numpy().tolist())
            loss_xyz = self.loss_iou3d(
                decode_bboxes3d_xyz,
                batch_bboxes3d_targets,
                avg_factor=num_batch_pos
            )
            loss_dim = self.loss_iou3d(
                decode_bboxes3d_dim,
                batch_bboxes3d_targets,
                avg_factor=num_batch_pos
            )
            loss_roty = self.loss_dir(
                reg_pos[:, 6:],
                batch_bboxes3d_targets[:, 6],
                avg_factor=num_batch_pos
            )
            mae_xyz = (decode_bboxes3d[:, :3] - batch_bboxes3d_targets[:, :3]).abs().mean()
            mae_dim = (decode_bboxes3d[:, 3:6] - batch_bboxes3d_targets[:, 3:6]).abs().mean()
            mae_roty = torch.acos(1 - loss_roty) / torch.pi * 180

        else:
            loss_xyz = reg_out.sum() * 0
            loss_dim = loss_xyz
            loss_roty = loss_xyz
            mae_xyz = loss_xyz
            mae_dim = loss_xyz
            mae_roty = loss_xyz

        return dict(
            loss_cls=loss_cls,
            loss_xyz=loss_xyz,
            loss_dim=loss_dim,
            loss_roty=loss_roty,
            mae_xyz=mae_xyz,
            mae_dim=mae_dim,
            mae_roty=mae_roty,
        )
