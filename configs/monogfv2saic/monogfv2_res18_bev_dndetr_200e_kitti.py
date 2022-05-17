_base_ = [
    '../_base_/models/monogfv2_res18_bev_dndetr.py',
    '../_base_/datasets/kitti-mono3d_car-saic.py',
    '../_base_/schedules/cyclic_200e_monocon.py',
    '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=10)

workflow = [('train', 1)]
log_config = dict(
    interval=20,
)
evaluation = dict(interval=10)
# fp16 = dict(loss_scale=512.)
runner = dict(type='EpochBasedRunner', max_epochs=200)
