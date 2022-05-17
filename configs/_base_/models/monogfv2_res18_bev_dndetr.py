model = dict(
    type='MonoSAIC',
    # pretrained=True,
    backbone=dict(
        type='ResNet',
        depth=18,
        norm_cfg=dict(type='BN'),
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')
    ),
    neck=dict(
        type='DilatedNeck',
        in_channels=[64, 128, 256, 512],
        out_channels=[64, 128, 64, 64],
        norm_cfg=dict(type='BN')),
    bbox_head=dict(
        type='BEVDNDETRMono3DHead',
        num_classes=1,
        in_channels=128,
        feat_channels=128,
        stacked_convs=3,
    ),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='ClassificationCost', weight=1.),
        ),
        dnassigner=dict(
            type='DenseAssigner3D',
            cls_cost=dict(type='ClassificationCost', weight=1.),
        ),
    ),
    test_cfg=dict())
