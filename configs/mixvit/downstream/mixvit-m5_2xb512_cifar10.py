_base_ = [\
    '../../_base_/models/mixvit_ds.py',
    '../../_base_/datasets/cifar10_bs16_224.py',
    '../../_base_/schedules/cifar10_bs128.py',
    '../../_base_/default_runtime.py',
]

randomness = dict(seed=0, diff_rank_seed=True) #seed setup

model_wrapper_cfg = dict(
                find_unused_parameters=True
            )

model = dict(
    backbone=dict(type='MixViT_tf', 
                  arch='m5',
                  init_cfg=dict(type='Pretrained', 
                  checkpoint='checkpoints/mixvit_m5.pth', 
                  prefix='backbone.'),
                  frozen_stages=-1
                  ), #, deploy=True), #deploy할때 사용할 것
    head=dict(
        num_classes=10,
        in_channels=384, # deploy=True #deploy할때 사용할 것
    ),
)

train_dataloader = dict(
    batch_size=512,
    num_workers=10,
    )

val_dataloader = dict(
    num_workers=10,
)

train_cfg = dict(_delete_=True, by_epoch=True, max_epochs=300, val_interval=1)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=1e-8),
    paramwise_cfg=dict(
        custom_keys={
            'attention_biases': dict(decay_mult=0.),
            'attention_bias_idxs': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=1e-06,
        type='LinearLR'),
    dict(begin=5, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]

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