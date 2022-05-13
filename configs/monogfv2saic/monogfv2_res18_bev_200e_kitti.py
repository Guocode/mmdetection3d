_base_ = [
    '../_base_/models/monogfv2_res18_bev.py',
    '../_base_/datasets/mini_kitti-mono3d_saic.py',
    '../_base_/schedules/cyclic_200e_monocon.py',
    '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=100)

workflow = [('train', 1)]
log_config = dict(
    interval=1,
)
evaluation = dict(interval=100)