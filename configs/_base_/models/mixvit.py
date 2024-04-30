# custom_imports = dict(imports=['MixViT'], allow_failed_imports=False)
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MixViT', arch='m0'),
    head=dict(
        type='EfficientViTClsHead',
        # type='LinearClsHead',
        num_classes=1000,
        in_channels=192,
    ),
    # init_cfg=dict(type='Pretrained', checkpoint='checkpoints/efficientvit/efficientvit_m3.pth')
)
