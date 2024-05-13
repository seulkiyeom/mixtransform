_base_ = [
    '../_base_/models/efficientvit.py',
    '../_base_/datasets/imagenet_bs64_mixtransform_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_mixtransform.py',
    '../_base_/default_runtime.py',
]

randomness = dict(seed=0, diff_rank_seed=True) #seed setup

model = dict(
    backbone=dict(arch='m5'),
    head=dict(
        in_channels=384,
    ),
    # init_cfg=dict(type='Pretrained', checkpoint='checkpoints/efficientvit/efficientvit_m3.pth')
)

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetRandomCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=512,
    dataset=dict(pipeline=train_pipeline)
)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
