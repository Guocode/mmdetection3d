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
        type='BEVMono3DHead',
        num_classes=3,
        in_channels=64,
        feat_channels=64,
        stacked_convs=3,
    ),
    train_cfg=None,
    test_cfg=dict(topk=30, local_maximum_kernel=3, max_per_img=30, thresh=0.4))
