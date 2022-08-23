# dataset settings
dataset_type = 'HuBMAPDataset'
data_root = 'data/hubmap'
img_norm_cfg = dict(
    mean=[196.869, 190.186, 194.802], std=[63.010, 66.765, 65.745], to_rgb=True)
image_size = (768, 768)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=image_size, multiscale_mode='value', keep_ratio=False),
    dict(type='HuBMAPAug'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train_images',
        ann_dir='train_masks',
        split='train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train_images',
        ann_dir='train_masks',
        split='val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train_images',
        ann_dir='train_masks',
        split='val.txt',
        pipeline=test_pipeline))
