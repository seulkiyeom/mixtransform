_base_ = [
    '../_base_/models/mixvit.py',
    '../_base_/datasets/imagenet_bs64_mixtransform_224.py', #'datasets/imagenet_bs256.py',
    '../_base_/schedules/imagenet_bs1024_adamw_mixtransform.py',  #'schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

randomness = dict(seed=0, diff_rank_seed=True) #seed setup

model_wrapper_cfg = dict(
                find_unused_parameters=True
            )

model = dict(
    backbone=dict(arch='m1'), #frozen_stages=-1), #, deploy=True), #deploy할때 사용할 것
    head=dict(
        in_channels=192, # deploy=True #deploy할때 사용할 것
    ),
    # init_cfg=dict(type='Pretrained', checkpoint='checkpoints/efficientvit/efficientvit_m1.pth')
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
    batch_size=2048,
    dataset=dict(pipeline=train_pipeline)
)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# runtime settings
default_hooks = dict(
    # save last three checkpoints
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',
        interval=20,
        max_keep_ckpts=3,
        rule='greater'))

# runtime setting (not working)
# custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (2 GPUs) x (2048 samples per GPU)
auto_scale_lr = dict(base_batch_size=2048)