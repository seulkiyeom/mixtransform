_base_ = [
    './mixvit-m0_2xb512_in1k.py',
]

model = dict(
    backbone=dict(arch='m2'),
    head=dict(
        in_channels=224,
    ),
    # init_cfg=dict(type='Pretrained', checkpoint='checkpoints/efficientvit/efficientvit_m0.pth')
)