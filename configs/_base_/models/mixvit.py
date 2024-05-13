# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MixViT', arch='m0'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MixViTClsHead',
        num_classes=1000,
        in_channels=192,
        topk=(1, 5),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_soft=True),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)