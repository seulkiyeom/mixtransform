_base_ = [
    './mixvit-m3_2xb512_in1k.py',
]

model = dict(
    backbone=dict(arch='m5'),
    head=dict(
        in_channels=384,
    ),
    # init_cfg=dict(type='Pretrained', checkpoint='checkpoints/efficientvit/efficientvit_m0.pth')
)