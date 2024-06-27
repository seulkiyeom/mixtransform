# for batch in each gpu is 256, 8 gpu
# lr = 1e-3 * 512 * 6 / 512 = 6e-3
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        weight_decay=0.025,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.attention_biases': dict(decay_mult=0.0),
        }),
    clip_grad=dict(max_norm=5.0),
)

# learning policy
param_scheduler = [
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=4e-6, by_epoch=True)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()