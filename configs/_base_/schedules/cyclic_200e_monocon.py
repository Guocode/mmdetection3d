lr = 1e-3
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.95, 0.99),
    weight_decay=0.01,
    paramwise_cfg=dict(bias_lr_mult=2., norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=100,
#     warmup_ratio=0.1,
#     step=[8, 11])
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=20)
