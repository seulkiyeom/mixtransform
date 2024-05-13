_base_ = [
    './mixvit-m1_2xb512_in1k.py',
]

model = dict(
    backbone=dict(arch='m0'),
    head=dict(
        in_channels=192,
    ),
    # init_cfg=dict(type='Pretrained', checkpoint='checkpoints/efficientvit/efficientvit_m0.pth')
)