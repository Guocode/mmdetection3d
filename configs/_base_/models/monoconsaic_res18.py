model = dict(
    type='MonoCon',
    # pretrained=True,
    backbone=dict(
        type='ResNet', depth=18, norm_cfg=dict(type='BN')),
    neck=dict(
        type='DilatedNeck',
        in_channels=[64, 128, 256, 512],
        out_channels=[64, 256, 256, 256],
        norm_cfg=dict(type='BN')),
    bbox_head=dict(
        type='MonoConSAICHead',
        in_channel=64,
        feat_channel=64,
        num_classes=3,
        loss_center_heatmap=dict(type='CenterNetGaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_center2kpt_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_kpt_heatmap=dict(type='CenterNetGaussianFocalLoss', loss_weight=1.0),
        loss_kpt_heatmap_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_dim=dict(type='DimAwareL1Loss', loss_weight=1.0),
        loss_depth=dict(type='LaplacianAleatoricUncertaintyLoss', loss_weight=1.0),
        loss_alpha_dir=dict(type='L1Loss', loss_weight=1.0),
        loss_iou2d=dict(type='GIoULoss', loss_weight=1.0),
        loss_iou3d=dict(type='GIoULoss', loss_weight=1.0),
    ),
    train_cfg=None,
    test_cfg=dict(topk=30, local_maximum_kernel=3, max_per_img=30, thresh=0.4))
