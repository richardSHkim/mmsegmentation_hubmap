_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/hubmap_aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=2),
    auxiliary_head=dict(in_channels=384, num_classes=2))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

evaluation = dict(interval=4000, metric='mDice', pre_eval=True)

batch_size = 8

run_name = 'upernet_swin_aug'
work_dirs = f'work_dirs/{run_name}'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='MlflowLoggerHook', 
            by_epoch=False, 
            exp_name='upernet_swin_hubmap',
            tags=dict({'mlflow.runName': run_name,
                        'lr_config': 'poly with warmup 40k', 
                        'image_size': (768, 768), 
                        'agumentation': 'default',
                        'classes': 'single-class', 
                        'batch_size': batch_size,
                      })
        ),
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=batch_size)
