_base_ = [
    # '../_base_/models/monocon_dla34.py',
    # '../_base_/datasets/mini_kitti-mono3d.py',
    '../_base_/schedules/cyclic_200e_monocon.py',
    '../_base_/default_runtime.py'
]
dataset_type = 'KittiMonoDepthDataset'
# data_root = 'data/mini_kitti/'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=False, use_camera=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(type='LoadDepthImageFromFile'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    # dict(type='Resize', img_scale=(1242, 375), keep_ratio=True),
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='NormIntrinsicByResizeShift',focal_length=710*0.25,norm_principal_point_offset=True,dst_size=(256,128)),
    # dict(type='PadBorders', size=(1248,384)),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d', 'gt_labels_3d',
            'centers2d', 'depths','kpts2d','kpts2d_valid','densedepth'
        ]),
]
test_pipeline = []
#     dict(type='LoadImageFromFileMono3D'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1242, 375),
#         flip=False,
#         transforms=[
#             dict(type='RandomFlip3D'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(
#                 type='DefaultFormatBundle3D',
#                 class_names=class_names,
#                 with_label=False),
#             dict(type='Collect3D', keys=['img']),
#         ])
# ]# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(type='LoadDepthImageFromFile'),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='NormIntrinsicByResizeShift', focal_length=710 * 0.25, norm_principal_point_offset=True,
    #      dst_size=(256, 128)),
    # dict(
    #     type='DefaultFormatBundle3D',
    #     class_names=class_names,
    #     with_label=False),
    # dict(type='Collect3D', keys=['img']),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 128),
        flip=False,
        transforms=[
            # dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img','densedepth']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_train_mono3d.coco.json',
        info_file=data_root + 'kitti_infos_train.pkl',
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
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'),
)
evaluation = dict(interval=2)

model = dict(
    type='MonoCon',
    pretrained=True,
    backbone=dict(
        type='DLANet', depth=34, norm_cfg=dict(type='BN')),
    neck=dict(
        type='DLANeck',
        in_channels=[16, 32, 64, 128, 256, 512],
        start_level=2,
        end_level=5,
        norm_cfg=dict(type='BN')),
    bbox_head=dict(
        type='DenseDepthHead',
        in_channel=64,
        feat_channel=64,
        densedepth_bin=1,
        loss_densedepth=dict(type='L1Loss')
    ),
    train_cfg=None,
    test_cfg=None)

checkpoint_config = dict(interval=5)

workflow = [('train', 1)]

depth_pretrain=True