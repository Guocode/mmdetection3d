model = dict(
    type='MonoSAIC',
    # pretrained=True,

    # backbone=dict(
    #     type='MobileNetV2',
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://mobilenet_v2')
    # ),
    backbone=dict(
        type='ResNet',
        depth=18,
        norm_cfg=dict(type='BN'),
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')
    ),
    neck=dict(
        type='DilatedNeck',
        in_channels=[64, 128, 256, 512],
        out_channels=[1, 1, 128, 128],
        ret_indices=[2,3],
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
            type='HungarianAssignerBEV',
        )
    ),
    test_cfg=dict())
