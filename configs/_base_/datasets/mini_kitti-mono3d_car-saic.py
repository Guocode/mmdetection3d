dataset_type = 'KittiMonoDataset'
data_root = 'data/mini_kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=False, use_camera=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    # dict(type='Resize', img_scale=(1242, 375), keep_ratio=True),
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='NormIntrinsic', focal_length=710 // 2, norm_principal_point_offset=True, dst_size=(1280 // 2, 384 // 2)),
    dict(type='AffineResize3D', dst_size=(1280 // 2, 384 // 2)),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d',
            'kpts2d', 'kpts2d_valid'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1248, 384),
        flip=False,
        transforms=[
            # dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='NormIntrinsic', focal_length=710//2, norm_principal_point_offset=True, dst_size=(1280 // 2, 384 // 2)),
            dict(type='AffineResize3D', dst_size=(1280 // 2, 384 // 2), affine_labels=False),
            # dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
# eval_pipeline = [
#     dict(type='LoadImageFromFileMono3D'),
#     dict(
#         type='DefaultFormatBundle3D',
#         class_names=class_names,
#         with_label=False),
#     dict(type='Collect3D', keys=['img'])
# ]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val_mono3d.coco.json',
        info_file=data_root + 'kitti_infos_val.pkl',
        img_prefix=data_root,
        classes=class_names,
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        box_type_3d='Camera'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val_mono3d.coco.json',
        info_file=data_root + 'kitti_infos_val.pkl',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val_mono3d.coco.json',
        info_file=data_root + 'kitti_infos_val.pkl',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'))
evaluation = dict(interval=2)
