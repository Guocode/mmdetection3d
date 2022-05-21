model = dict(
    type='MonoSAIC',
    # pretrained=True,
    backbone=dict(
        type='ResNet', depth=18, norm_cfg=dict(type='BN')),
    neck=dict(
        type='DilatedNeck',
        in_channels=[64, 128, 256, 512],
        out_channels=[256, 256, 256, 256],
        norm_cfg=dict(type='BN')),
    bbox_head=dict(
        type='MonoGFocalV2SAICHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        strides=[4, 8, 16, 32],
        stacked_convs=1,
        loss_cls=dict(
            type='QualityFocalLoss',
            activated=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        reg_topk=4,
        reg_channels=64,
        add_mean=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)
    ),
    train_cfg=dict(
        # assigner=dict(type='ATSSAssigner', topk=9)
        assigner=dict(
            type='SimOTAAssigner',
            center_radius=2.5,
            candidate_topk=7,
            iou_weight=3.0,
            cls_weight=1.0)
    ),
    test_cfg=dict(topk=30, local_maximum_kernel=3, max_per_img=30, thresh=0.4))
