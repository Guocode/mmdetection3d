_base_ = [
    '../_base_/models/monoconsaic_res18.py',
    '../_base_/datasets/mini_kitti-mono3d.py',
    '../_base_/schedules/cyclic_200e_monocon.py',
    '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=5)

workflow = [('train', 1)]
