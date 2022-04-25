# Copyright (c) OpenMMLab. All rights reserved.

# import mmcv
import numpy as np
import torch

from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes,
                               DepthInstance3DBoxes, LiDARInstance3DBoxes)
from mmdet3d.models.builder import build_head
from mmdet3d.models.losses.rtm3d_losses import _transpose_and_gather_feat


def test_rtm3d_mono3d_head():
    head_cfg = dict(
        type='RTM3DHead',
        in_channel=256,
        feat_channel=64,
        num_classes=3,
        loss_hm=dict(type='RTM3DFocalLoss'),
        loss_hm_hp=dict(type='RTM3DFocalLoss'),
        loss_kp=dict(type='RegWeightedL1Loss'),
        loss_reg=dict(type='RegL1Loss'),
        loss_rot=dict(type='BinRotLoss'),
        loss_position=dict(type='Position_loss', feat_w=312, feat_h=96),
    )
    head_model = build_head(head_cfg)
    head_model.cuda()

    # assert len(ret_dict) == 2
    # assert len(ret_dict[0]) == 1
    # assert ret_dict[0][0].shape == torch.Size([2, 3, 32, 32])
    # assert ret_dict[1][0].shape == torch.Size([2, 8, 32, 32])
    # test loss

    feats = torch.rand([1, 256, 96, 312], dtype=torch.float32).cuda()
    gt_bboxes = [
        torch.Tensor([[722.4446, 151.0021, 832.2931, 314.5869]]).cuda(),
    ]
    gt_bboxes_3d = [
        CameraInstance3DBoxes(torch.Tensor([[1.9047, 1.4695, 8.4150, 1.2000, 1.8900, 0.4800, 0.0100]]), box_dim=7),
    ]
    gt_labels = [torch.Tensor([1]).cuda() for i in range(1)]
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
            cam2img=torch.asarray([[7.070493e+02, 0.000000e+00, 6.160814e+02, 4.575831e+01],
                                [0.000000e+00, 7.070493e+02, 1.875066e+02, - 3.454157e-01],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03], ]),
            scale_factor=np.array([1., 1., 1., 1.], dtype=np.float32),
            pad_shape=[128, 128],
            trans_mat=np.array([[0.25, 0., 0.], [0., 0.25, 0], [0., 0., 1.]],
                               dtype=np.float32),
            affine_aug=False,
            img_shape=(384, 1248, 3),
            box_type_3d=CameraInstance3DBoxes) for i in range(1)
    ]

    print('box',gt_bboxes)
    # test forward
    ret_dict = head_model([feats])
    targets = head_model.get_targets(gt_bboxes, gt_labels, gt_bboxes_3d,
                       centers2d, depths, kpts2d,
                       kpts2d_valid,feats.shape, img_metas)
    pgt_wh = torch.zeros((1,2,96,312),dtype=torch.float32).cuda()
    pgt_wh[:,:,18290//312,18290%312] = targets['wh'][:,0,:]
    pgt_hps = torch.zeros((1,18,96,312),dtype=torch.float32).cuda()
    pgt_hps[:,:,18290//312,18290%312] = targets['hps'][:,0,:]#.reshape(9,2).T.reshape(-1)
    pgt_rot = torch.zeros((1,8,96,312),dtype=torch.float32).cuda()
    pgt_rot[:,:,18290//312,18290%312] = torch.asarray([0,1,0.9775,0.2110,0,1,-0.9775,-0.2110])
    pgt_dim = torch.zeros((1,3,96,312),dtype=torch.float32).cuda()
    pgt_dim[:,:,18290//312,18290%312] = targets['dim'][:,0,:]
    pgt_prob = torch.zeros((1, 1, 96, 312), dtype=torch.float32).cuda()
    pgt_prob[:,:,18290//312,18290%312] = 1
    pgt_reg = torch.zeros((1, 2, 96, 312), dtype=torch.float32).cuda()
    pgt_reg[:, :, 18290 // 312, 18290 % 312] = targets['reg'][:,0,:]
    pgt_hp_offset = torch.zeros((1, 2, 96*312), dtype=torch.float32).cuda()
    pgt_hp_offset[:,:,[17977, 11726, 12036, 23892, 24518, 11752, 12061, 23917, 24544]] = targets['hp_offset'][:,:9,:].reshape((9,2)).T
    pgt_hp_offset = torch.reshape(pgt_hp_offset,(1,2,96,312))
    ret_dict = [-torch.log((1 / (targets['hm'] + 1e-12)) - 1),pgt_wh,pgt_hps,pgt_rot,pgt_dim,pgt_prob,pgt_reg,-torch.log((1 / (targets['hm_hp'] + 1e-12)) - 1),pgt_hp_offset]

    loss = head_model.loss(*ret_dict, gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d,
                       centers2d, depths, kpts2d,
                       kpts2d_valid, img_metas)
    print(loss)

    loss = loss['hm_loss'] + loss['wh_loss'] +  loss['off_loss']  + loss['hp_loss'] + loss['hp_offset_loss'] +loss['hm_hp_loss']  + loss['dim_loss'] + loss['rot_loss']  + loss['prob_loss'] + loss['coor_loss']
    print(loss,feats.mean())
    results = head_model.get_bboxes(*ret_dict, img_metas,rescale=True)
    # assert len(results) == 2
    # assert len(results[0]) == 4
    for result in results[0]:
        print(result)

if __name__ == '__main__':
    test_rtm3d_mono3d_head()
