_base_ = [
    '../_base_/models/monogfv2_res18_bev_evendz.py',
    '../_base_/datasets/kitti-mono3d_car-saic.py',
    # '../_base_/datasets/mini_kitti-mono3d_car-saic.py',
    '../_base_/schedules/cyclic_200e_monocon.py',
    # '../_base_/schedules/mmdet_schedule_1x.py',

    '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=5)

workflow = [('train', 1)]
log_config = dict(
    interval=20,
)
evaluation = dict(interval=5)
# fp16 = dict(loss_scale=512.)
runner = dict(type='EpochBasedRunner', max_epochs=72)
