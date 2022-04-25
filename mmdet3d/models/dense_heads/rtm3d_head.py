import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils.gaussian_target import (gaussian_radius, gen_gaussian_target)
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)

INF = 1e8
EPS = 1e-12
PI = np.pi


@HEADS.register_module()
class RTM3DHead(nn.Module):
    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 bbox3d_code_size=7,
                 num_joints=9,
                 num_alpha_bins=12,
                 max_objs=30,
                 vector_regression_level=1,
                 pred_bbox2d=True,
                 loss_hm=None,
                 loss_hm_hp=None,
                 loss_kp=None,
                 loss_reg=None,
                 loss_rot=None,
                 loss_position=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 ):
        super(RTM3DHead, self).__init__()
        assert bbox3d_code_size >= 7
        self.num_classes = num_classes
        self.bbox_code_size = bbox3d_code_size
        self.pred_bbox2d = pred_bbox2d
        self.max_objs = max_objs
        self.num_alpha_bins = num_alpha_bins
        self.num_joints = num_joints
        self.vector_regression_level = vector_regression_level

        self.hm_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.hps_head = self._build_head(in_channel, feat_channel, 18)
        self.rot_head = self._build_head(in_channel, feat_channel, 8)
        self.dim_head = self._build_head(in_channel, feat_channel, 3)
        self.prob_head = self._build_head(in_channel, feat_channel, 1)
        self.reg_head = self._build_head(in_channel, feat_channel, 2)
        self.hm_hp_head = self._build_head(in_channel, feat_channel, 9)
        self.hm_offset_head = self._build_head(in_channel, feat_channel, 2)

        self.crit_hm = build_loss(loss_hm)
        self.crit_hm_hp = build_loss(loss_hm_hp)
        self.crit_kp = build_loss(loss_kp)
        self.crit_reg = build_loss(loss_reg)
        self.crit_rot = build_loss(loss_rot)
        self.position_loss= build_loss(loss_position)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        # self.const = self.const.to(self.device)

    def _build_head(self, in_channel, feat_channel, out_channel):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    # def init_weights(self):
    #     bias_init = bias_init_with_prob(0.1)
    #     self.heatmap_head[-1].bias.data.fill_(bias_init)  # -2.19
    #     self.kpt_heatmap_head[-1].bias.data.fill_(bias_init)
    #     for head in [self.wh_head, self.offset_head, self.center2kpt_offset_head, self.depth_head,
    #                  self.kpt_heatmap_offset_head, self.dim_head, self.dir_feat,
    #                  self.dir_cls, self.dir_reg]:
    #         for m in head.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 normal_init(m, std=0.001)

    def forward(self, feats):
        return self.forward_single(feats[0])

    def forward_single(self, feat):
        hm_preds = self.hm_head(feat)
        wh_preds = self.wh_head(feat)
        hps_preds = self.hps_head(feat)
        rot_preds = self.rot_head(feat)
        dim_preds = self.dim_head(feat)
        prob_preds = self.prob_head(feat)
        reg_preds = self.reg_head(feat)
        hm_hp_preds = self.hm_hp_head(feat)
        hp_offset_preds = self.hm_offset_head(feat)
        return hm_preds, wh_preds, hps_preds, rot_preds, dim_preds, prob_preds, reg_preds, hm_hp_preds, hp_offset_preds

    @force_fp32(apply_to=('hm_preds', 'wh_preds', 'hps_preds', 'rot_preds',
                          'dim_preds', 'prob_preds', 'reg_preds', 'hm_hp_preds',
                          'hp_offset_preds'))
    def loss(self,
             hm_preds,
             wh_preds,
             hps_preds,
             rot_preds,
             dim_preds,
             prob_preds,
             reg_preds,
             hm_hp_preds,
             hp_offset_preds,
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

        # assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) \
        #        == len(center2kpt_offset_preds) == len(kpt_heatmap_preds) == len(kpt_heatmap_offset_preds) \
        #        == len(dim_preds) == len(alpha_cls_preds) == len(alpha_offset_preds) == 1

        hm_pred = hm_preds
        wh_pred = wh_preds
        hps_pred = hps_preds
        rot_pred = rot_preds
        dim_pred = dim_preds
        prob_pred = prob_preds
        reg_pred = reg_preds
        hm_hp_pred = hm_hp_preds
        hp_offset_pred = hp_offset_preds

        batch_size = hm_pred.shape[0]

        target_result = self.get_targets(gt_bboxes, gt_labels,
                                         gt_bboxes_3d,
                                         centers2d,
                                         depths,
                                         gt_kpts_2d,
                                         gt_kpts_valid_mask,
                                         hm_pred.shape,
                                         img_metas)

        hm_pred = torch.clamp(torch.sigmoid(hm_pred), min=1e-7, max=1-1e-7)
        hm_hp_pred = torch.clamp(torch.sigmoid(hm_hp_pred), min=1e-7, max=1 - 1e-7)
        hm_loss = self.crit_hm(hm_pred, target_result['hm'])
        hp_loss = self.crit_kp(hps_pred, target_result['hps_mask'], target_result['ind'], target_result['hps'],
                               target_result['dep'])
        wh_loss = self.crit_reg(wh_pred, target_result['reg_mask'], target_result['ind'], target_result['wh'])
        dim_loss = self.crit_reg(dim_pred, target_result['reg_mask'], target_result['ind'], target_result['dim'])
        rot_loss = self.crit_rot(rot_pred, target_result['rot_mask'], target_result['ind'], target_result['rotbin'],
                                 target_result['rotres'])
        off_loss = self.crit_reg(reg_pred, target_result['reg_mask'], target_result['ind'], target_result['reg'])
        hp_offset_loss = self.crit_reg(hp_offset_pred, target_result['hp_mask'], target_result['hp_ind'],
                                       target_result['hp_offset'])
        hm_hp_loss = self.crit_hm_hp(hm_hp_pred, target_result['hm_hp'])
        #transform to rtm3d order
        coor_loss, prob_loss, box_score = self.position_loss(dim_pred, rot_pred, prob_pred, hps_pred, target_result)
        loss_stats = dict(box_score=box_score, hm_loss= hm_loss, hp_loss= hp_loss,
                      hm_hp_loss=hm_hp_loss, hp_offset_loss=hp_offset_loss,
                      wh_loss=wh_loss, off_loss=off_loss, dim_loss=dim_loss,
                      rot_loss=rot_loss, prob_loss=prob_loss, coor_loss=coor_loss)
        return loss_stats
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_dim=loss_dim,
            loss_center2kpt_offset=loss_center2kpt_offset,
            loss_kpt_heatmap=loss_kpt_heatmap,
            loss_kpt_heatmap_offset=loss_kpt_heatmap_offset,
            loss_alpha_cls=loss_alpha_cls,
            loss_alpha_reg=loss_alpha_reg,
            loss_depth=loss_depth,
        )

    def get_targets(self, gt_bboxes, gt_labels,
                    gt_bboxes_3d,
                    centers2d,
                    depths,
                    gt_kpts_2d,
                    gt_kpts_valid_mask,
                    feat_shape,
                    img_metas):
        img_h, img_w, _ = img_metas[0]['img_shape']
        bs, _, feat_h, feat_w = feat_shape
        scale2feat = feat_h / img_h
        hm = torch.zeros((bs, self.num_classes, feat_h, feat_w), dtype=torch.float32).to(gt_bboxes[0].device)
        hm_hp = torch.zeros((bs, 9, feat_h, feat_w), dtype=torch.float32).to(gt_bboxes[0].device)
        wh = torch.zeros((bs, self.max_objs, 2), dtype=torch.float32).to(gt_bboxes[0].device)
        dim = torch.zeros((bs, self.max_objs, 3), dtype=torch.float32).to(gt_bboxes[0].device)
        location = torch.zeros((bs, self.max_objs, 3), dtype=torch.float32).to(gt_bboxes[0].device)
        dep = torch.zeros((bs, self.max_objs, 1), dtype=torch.float32).to(gt_bboxes[0].device)
        ori = torch.zeros((bs, self.max_objs, 1), dtype=torch.float32).to(gt_bboxes[0].device)
        rotbin = torch.zeros((bs, self.max_objs, 2), dtype=torch.int64).to(gt_bboxes[0].device)
        rotres = torch.zeros((bs, self.max_objs, 2), dtype=torch.float32).to(gt_bboxes[0].device)
        rot_mask = torch.zeros((bs, self.max_objs), dtype=torch.uint8).to(gt_bboxes[0].device)
        hps = torch.zeros((bs, self.max_objs, self.num_joints * 2), dtype=torch.float32).to(gt_bboxes[0].device)
        hps_cent = torch.zeros((bs, self.max_objs, 2), dtype=torch.float32).to(gt_bboxes[0].device)
        reg = torch.zeros((bs, self.max_objs, 2), dtype=torch.float32).to(gt_bboxes[0].device)
        ind = torch.zeros((bs, self.max_objs), dtype=torch.int64).to(gt_bboxes[0].device)
        reg_mask = torch.zeros((bs, self.max_objs), dtype=torch.uint8).to(gt_bboxes[0].device)
        inv_mask = torch.zeros((bs, self.max_objs, self.num_joints * 2), dtype=torch.uint8).to(gt_bboxes[0].device)
        hps_mask = torch.zeros((bs, self.max_objs, self.num_joints * 2), dtype=torch.uint8).to(gt_bboxes[0].device)
        coor_kps_mask = torch.zeros((bs, self.max_objs, self.num_joints * 2), dtype=torch.uint8).to(gt_bboxes[0].device)
        hp_offset = torch.zeros((bs, self.max_objs * self.num_joints, 2), dtype=torch.float32).to(gt_bboxes[0].device)
        hp_ind = torch.zeros((bs, self.max_objs * self.num_joints), dtype=torch.int64).to(gt_bboxes[0].device)
        hp_mask = torch.zeros((bs, self.max_objs * self.num_joints), dtype=torch.int64).to(gt_bboxes[0].device)
        rot_scalar = torch.zeros((bs, self.max_objs, 1), dtype=torch.float32).to(gt_bboxes[0].device)
        opinv = torch.zeros((bs,2,3),dtype=torch.float32).to(gt_bboxes[0].device)
        opinv[:,0,0],opinv[:,1,1] = 1./scale2feat,1./scale2feat
        calib = torch.zeros((bs,3,4),dtype=torch.float32).to(gt_bboxes[0].device)
        for i in range(bs):
            gt_bboxes_scaled = gt_bboxes[i] * scale2feat  # scale to feature size
            gt_kpts_2d_scaled = gt_kpts_2d[i] * scale2feat
            calib[i] = torch.tensor(img_metas[i]['cam2img'][:3])
            # img_metas[i]['trans_mat'] = torch.asarray([[scale2feat, 0., 0.], [0., scale2feat, 0], [0., 0., 1.]],
            #                    dtype=torch.float32)
            # img_metas[i]['trans_output_inv'] = torch.asarray([[1./scale2feat, 0., 0.], [0., 1./scale2feat, 0], [0., 0., 1.]],
            #                    dtype=torch.float32)
            for k in range(len(gt_labels[i])):
                alpha = gt_bboxes_3d[i].local_yaw[k]
                if alpha < torch.pi / 6. or alpha > 5 * torch.pi / 6.:
                    rotbin[i, k, 0] = 1
                    rotres[i, k, 0] = alpha - (-0.5 * np.pi)
                if alpha > -torch.pi / 6. or alpha < -5 * torch.pi / 6.:
                    rotbin[i, k, 1] = 1
                    rotres[i, k, 1] = alpha - (0.5 * np.pi)
                rot_scalar[i, k] = alpha
                box_h = (gt_bboxes_scaled[k, 3] - gt_bboxes_scaled[k, 1])
                box_w = (gt_bboxes_scaled[k, 2] - gt_bboxes_scaled[k, 0])
                radius = gaussian_radius([box_h, box_w],min_overlap=0.3)
                radius = max(0, int(radius))
                ct = torch.asarray(
                    ((gt_bboxes_scaled[k, 0] + gt_bboxes_scaled[k, 2]) / 2, (gt_bboxes_scaled[k, 1] + gt_bboxes_scaled[k, 3]) / 2)).to(
                    gt_bboxes[0].device)
                ct_int = ct.int()
                wh[i, k] = torch.asarray((1. * box_w, 1. * box_h))
                ind[i, k] = ct_int[1] * feat_w + ct_int[0]
                reg[i, k] = ct - ct_int
                dim[i, k] = gt_bboxes_3d[i].dims[k]
                dep[i, k] = depths[i][k]
                ori[i, k] = gt_bboxes_3d[i].yaw[k]
                location[i, k] = gt_bboxes_3d[i].center[k]
                reg_mask[i, k] = 1
                # num_kpts = pts[:, 2].sum()
                # if num_kpts == 0:
                #     hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                #     reg_mask[k] = 0
                rot_mask[i, k] = 1
                hp_radius = radius
                hp_radius = max(0, int(hp_radius))
                hps_cent[i, k, :] = ct
                for j in range(self.num_joints):
                    hps[i, k, j * 2:j * 2 + 2] = gt_kpts_2d_scaled[k, j] - ct_int
                    hps_mask[i, k, j * 2: j * 2 + 2] = 1  # (gt_kpts_valid_mask[i][k,j * 2: j * 2 + 2]==1).uint8()
                    if gt_kpts_valid_mask[i][k, j] == 1:
                        pt_int = gt_kpts_2d_scaled[k, j].int()
                        is_pt_inside_image = (0 <= pt_int[0] < feat_w) and (0 <= pt_int[1] < feat_h)
                        if not is_pt_inside_image:
                            continue
                        inv_mask[i, k, j * 2: j * 2 + 2] = 1
                        coor_kps_mask[i, k, j * 2: j * 2 + 2] = 1
                        hp_offset[i, k * self.num_joints + j] = gt_kpts_2d_scaled[k, j] - pt_int
                        hp_ind[i, k * self.num_joints + j] = pt_int[1] * feat_w + pt_int[0]
                        hp_mask[i, k * self.num_joints + j] = 1

                        gen_gaussian_target(hm_hp[i, j],
                                            [pt_int[0], pt_int[1]], hp_radius)
                if coor_kps_mask[i, k, 16] == 0 or coor_kps_mask[i, k, 17] == 0:
                    coor_kps_mask[i, k, :] = coor_kps_mask[i, k, :] * 0
                is_ct_inside_image = (0 <= ct_int[0] < feat_w) and (0 <= ct_int[1] < feat_h)
                if not is_ct_inside_image:
                    continue
                gen_gaussian_target(hm[i, gt_labels[i][k].int()], [ct_int[0], ct_int[1]], radius)
        target_result = dict(
            hm=hm,
            reg_mask=reg_mask,
            ind=ind,
            wh=wh,
            hps=hps,
            hps_mask=hps_mask,
            dim=dim,
            rotbin=rotbin,
            rotres=rotres,
            rot_mask=rot_mask,
            dep=dep,
            rot_scalar=rot_scalar,
            hps_cent=hps_cent,
            location=location,
            ori=ori,
            coor_kps_mask=coor_kps_mask,
            inv_mask=inv_mask,
            reg=reg,
            hm_hp=hm_hp,
            hp_offset=hp_offset,
            hp_ind=hp_ind,
            hp_mask=hp_mask,
            opinv=opinv,
            calib=calib
        )
        return target_result

    def get_bboxes(self,hm_preds, wh_preds, hps_preds, rot_preds, dim_preds, prob_preds, reg_preds, hm_hp_preds, hp_offset_preds,  meta=None,K=100,thresh=0.4,rescale=False):
        # hm = hm_preds[0]
        # kps = hps_preds[0]
        # reg = reg_preds[0]
        # wh = wh_preds[0]
        # dim = dim_preds[0]
        # rot = rot_preds[0]
        # prob = prob_preds[0]
        # hm_hp = hm_hp_preds[0]
        # hp_offset = hp_offset_preds[0]
        for bs in range(hm_preds.shape[0]):
            meta[bs]['trans_mat'] = (torch.eye(3,dtype=torch.float32)*hm_preds.shape[2]/meta[bs]['img_shape'][0]).to(hm_preds.device)
            meta[bs]['trans_mat_inv'] = (torch.eye(3,dtype=torch.float32)/hm_preds.shape[2]*meta[bs]['img_shape'][0]).to(hm_preds.device)
        hm = torch.sigmoid(hm_preds)
        kps = hps_preds
        reg = reg_preds
        wh = wh_preds
        dim = dim_preds
        rot = rot_preds
        prob = prob_preds
        hm_hp = torch.sigmoid(hm_hp_preds)
        hp_offset = hp_offset_preds

        batch, cat, height, width = hm.shape
        num_joints = kps.shape[1] // 2
        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        # hm_show,_=torch.max(hm_hp,1)
        # hm_show=hm_show.squeeze(0)
        # hm_show=hm_show.detach().cpu().numpy().copy()
        # plt.imshow(hm_show, 'gray')
        # plt.show()

        hm = get_local_maximum(hm)
        scores, inds, clses, ys, xs = get_topk_from_heatmap(hm, k=K)

        kps = transpose_and_gather_feat(kps, inds)
        kps = kps.view(batch, K, num_joints * 2)
        kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
        kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
        if reg is not None:
            reg = transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        clses = clses.view(batch, K)
        scores = scores.view(batch, K)

        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        dim = transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, K, 3)
        # dim[:, :, 0] = torch.exp(dim[:, :, 0]) * 1.63
        # dim[:, :, 1] = torch.exp(dim[:, :, 1]) * 1.53
        # dim[:, :, 2] = torch.exp(dim[:, :, 2]) * 3.88
        rot = transpose_and_gather_feat(rot, inds)
        rot = rot.view(batch, K, 8)
        if prob is not None:
            prob = transpose_and_gather_feat(prob, inds)[:, :, 0]
            prob = prob.view(batch, K, 1)
        if hm_hp is not None:
            hm_hp = get_local_maximum(hm_hp)
            thresh = 0.25
            kps = kps.view(batch, K, num_joints, 2).permute(
                0, 2, 1, 3).contiguous()  # b x J x K x 2
            reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
            hm_score, hm_inds, hm_ys, hm_xs = [], [], [], []
            for i in range(self.num_joints):
                _hm_score, _hm_inds, _, _hm_ys, _hm_xs = get_topk_from_heatmap(hm_hp[:, i:i + 1], k=K)  # b x J x K
                hm_score.append(_hm_score)
                hm_inds.append(_hm_inds)
                hm_ys.append(_hm_ys)
                hm_xs.append(_hm_xs)
            hm_score, hm_inds, hm_ys, hm_xs = \
                torch.stack(hm_score, 1), torch.stack(hm_inds, 1), torch.stack(hm_ys,1), torch.stack(hm_xs, 1)


            if hp_offset is not None:
                hp_offset = transpose_and_gather_feat(
                    hp_offset, hm_inds.view(batch, -1))
                hp_offset = hp_offset.view(batch, num_joints, K, 2)
                hm_xs = hm_xs + hp_offset[:, :, :, 0]
                hm_ys = hm_ys + hp_offset[:, :, :, 1]
            else:
                hm_xs = hm_xs + 0.5
                hm_ys = hm_ys + 0.5
            mask = (hm_score > thresh).float()
            hm_score = (1 - mask) * -1 + mask * hm_score
            hm_ys = (1 - mask) * (-10000) + mask * hm_ys
            hm_xs = (1 - mask) * (-10000) + mask * hm_xs
            hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
                2).expand(batch, num_joints, K, K, 2)
            dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
            min_dist, min_ind = dist.min(dim=3)  # b x J x K
            hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
            min_dist = min_dist.unsqueeze(-1)
            min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
                batch, num_joints, K, 1, 2)
            hm_kps = hm_kps.gather(3, min_ind)
            hm_kps = hm_kps.view(batch, num_joints, K, 2)
            l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
                   (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
                   (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
            mask = (mask > 0).float().expand(batch, num_joints, K, 2)
            kps = (1 - mask) * hm_kps + mask * kps
            kps = kps[:,[7,8,4,3,6,5,1,2,0],:,:]#transform to KM3D order
            kps = kps.permute(0, 2, 1, 3).contiguous().view(
                batch, K, num_joints * 2)
            hm_score = hm_score.permute(0, 2, 1, 3).squeeze(3).contiguous()
        position, rot_y, kps_inv = self.gen_position(kps, dim[:,:,[1,2,0]], rot, meta)

        # detections = torch.cat([bboxes, scores, kps_inv, dim, hm_score, rot_y, position, prob, clses], dim=2)
        box_type_3d = meta[0]['box_type_3d']
        det_results = []
        for i in range(batch):
            mask = (scores[i] >thresh)
            bboxes = bboxes[i][mask] * meta[i]['trans_mat_inv'][0, 0] if rescale else bboxes[i][mask]
            bboxes = torch.cat([bboxes, scores[i][mask].unsqueeze(-1)], dim=-1)
            det_results.append(
            [box_type_3d(torch.cat([position[i][mask],dim[i][mask],rot_y[i][mask]],dim=1),
                         box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5)),
             scores[i][mask],
             clses[i][mask],
             bboxes
             ]
            )
        return det_results

    def gen_position(self,kps, dim, rot, meta):
        const = torch.Tensor(
          [[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
           [-1, 0], [0, -1], [-1, 0], [0, -1]])
        const = const.unsqueeze(0).unsqueeze(0)
        b,c,_ = kps.shape
        opinv = torch.stack([mt['trans_mat_inv'][:2] for mt in meta]).to(kps.device)
        calib = torch.stack([mt['cam2img'] for mt in meta]).to(kps.device)

        opinv = opinv.unsqueeze(1)
        opinv = opinv.expand(b, c, -1, -1).contiguous().view(-1, 2, 3).float()
        kps = kps.view(b, c, -1, 2).permute(0, 1, 3, 2)
        hom = torch.ones(b, c, 1, 9).to(kps.device)
        kps = torch.cat((kps, hom), dim=2).view(-1, 3, 9)
        kps = torch.bmm(opinv, kps).view(b, c, 2, 9)
        kps = kps.permute(0, 1, 3, 2).contiguous().view(b, c, -1)  # 16.32,18
        si = torch.zeros_like(kps[:, :, 0:1]) + calib[:, 0:1, 0:1]
        alpha_idx = rot[:, :, 1] > rot[:, :, 5]
        alpha_idx = alpha_idx.float()
        alpha1 = torch.atan2(rot[:, :, 2] , rot[:, :, 3]) + (-0.5 * np.pi)
        alpha2 = torch.atan2(rot[:, :, 6] , rot[:, :, 7]) + (0.5 * np.pi)
        alpna_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx)
        alpna_pre = alpna_pre.unsqueeze(2)
        # alpna_pre=rot_gt

        rot_y = alpna_pre + torch.atan2(kps[:, :, 16:17] - calib[:, 0:1, 2:3], si)
        rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi
        rot_y[rot_y < - np.pi] = rot_y[rot_y < - np.pi] + 2 * np.pi

        calib = calib.unsqueeze(1)
        calib = calib.expand(b, c, -1, -1).contiguous()
        kpoint = kps[:, :, :16]
        f = calib[:, :, 0, 0].unsqueeze(2)
        f = f.expand_as(kpoint)
        cx, cy = calib[:, :, 0, 2].unsqueeze(2), calib[:, :, 1, 2].unsqueeze(2)
        cxy = torch.cat((cx, cy), dim=2)
        cxy = cxy.repeat(1, 1, 8)  # b,c,16
        kp_norm = (kpoint - cxy) / f

        l = dim[:, :, 2:3]
        h = dim[:, :, 0:1]
        w = dim[:, :, 1:2]
        cosori = torch.cos(rot_y)
        sinori = torch.sin(rot_y)

        B = torch.zeros_like(kpoint)
        C = torch.zeros_like(kpoint)

        kp = kp_norm.unsqueeze(3)  # b,c,16,1
        const = const.expand(b, c, -1, -1).to(kp.device)
        A = torch.cat([const, kp], dim=3)

        B[:, :, 0:1] = l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, 1:2] = h * 0.5
        B[:, :, 2:3] = l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, 3:4] = h * 0.5
        B[:, :, 4:5] = -l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, 5:6] = h * 0.5
        B[:, :, 6:7] = -l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, 7:8] = h * 0.5
        B[:, :, 8:9] = l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, 9:10] = -h * 0.5
        B[:, :, 10:11] = l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, 11:12] = -h * 0.5
        B[:, :, 12:13] = -l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, 13:14] = -h * 0.5
        B[:, :, 14:15] = -l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, 15:16] = -h * 0.5

        C[:, :, 0:1] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 1:2] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 2:3] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 3:4] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 4:5] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 5:6] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 6:7] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 7:8] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 8:9] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 9:10] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 10:11] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 11:12] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 12:13] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 13:14] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 14:15] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 15:16] = l * 0.5 * sinori + w * 0.5 * cosori

        B = B - kp_norm * C

        # A=A*kps_mask1

        AT = A.permute(0, 1, 3, 2)
        AT = AT.view(b * c, 3, 16)
        A = A.view(b * c, 16, 3)
        B = B.view(b * c, 16, 1).float()
        # mask = mask.unsqueeze(2)

        pinv = torch.bmm(AT, A)
        pinv = torch.inverse(pinv)  # b*c 3 3

        pinv = torch.bmm(pinv, AT)
        pinv = torch.bmm(pinv, B)
        pinv = pinv.view(b, c, 3, 1).squeeze(3)

        # pinv[:, :, 1] = pinv[:, :, 1] + dim[:, :, 0] / 2
        return pinv, rot_y, kps

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
