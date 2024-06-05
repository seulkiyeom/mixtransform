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

train_dataloader = dict(
    batch_size=512
    )

# runtime settings
default_hooks = dict(
    # save last three checkpoints
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',
        interval=20,
        max_keep_ckpts=3,
        rule='greater'))

# # runtime setting
# custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (6 GPUs) x (512 samples per GPU)
auto_scale_lr = dict(base_batch_size=3072)