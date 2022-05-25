_base_ = [
    '../_base_/models/monodetr3d_res18.py',
    '../_base_/datasets/kitti-mono3d_car-saic.py',
    # '../_base_/datasets/mini_kitti-mono3d_saic.py',
    '../_base_/schedules/cyclic_200e_monocon.py',
    '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=5)

workflow = [('train', 1)]
log_config = dict(
    interval=10,
)
evaluation = dict(interval=5)
runner = dict(type='EpochBasedRunner', max_epochs=72)

data = dict(
    samples_per_gpu=4,)