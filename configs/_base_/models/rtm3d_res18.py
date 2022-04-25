model = dict(
    type='RTM3DMono3D',
    # pretrained=True,
    backbone=dict(
        type='ResNet', depth=18, norm_cfg=dict(type='BN')),
    # neck=dict(
    #     type='DLAUp',
    #     in_channels_list=[64, 128, 256, 512],
    #     scales_list=(1, 2, 4, 8),
    #     start_level=2,
    #     norm_cfg=dict(type='BN')),
    neck=dict(
        type='RTM3DNeck',
        in_channels=[512],
        start_level=2,
        end_level=5,
        norm_cfg=dict(type='BN')),
    bbox_head=dict(
        type='RTM3DHead',
        in_channel=256,
        feat_channel=64,
        num_classes=3,
        loss_hm=dict(type='RTM3DFocalLoss'),
        loss_hm_hp=dict(type='RTM3DFocalLoss'),
        loss_kp=dict(type='RegWeightedL1Loss'),
        loss_reg=dict(type='RegL1Loss'),
        loss_rot=dict(type='BinRotLoss'),
        loss_position=dict(type='Position_loss',feat_w=320,feat_h=96),
    ),
    train_cfg=None,
    test_cfg=dict(topk=30, local_maximum_kernel=3, max_per_img=30, thresh=0.4))
