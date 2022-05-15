model = dict(
    type='MonoSAIC',
    # pretrained=True,
    backbone=dict(
        type='ResNet', depth=18, norm_cfg=dict(type='BN')),
    neck=dict(
        type='DilatedNeck',
        in_channels=[64, 128, 256, 512],
        out_channels=[64, 64, 64, 64],
        norm_cfg=dict(type='BN')),
    bbox_head=dict(
        type='BEVDETRMono3DHead',
        num_classes=3,
        in_channels=64,
        feat_channels=64,
        stacked_convs=3,
    ),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='ClassificationCost', weight=1.),
        ),
    ),
    test_cfg=dict())
