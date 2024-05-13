# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientViT', arch='m0'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MixViTClsHead',
        num_classes=1000,
        in_channels=192,
        topk=(1, 5),
    ),
)