# Copyright (c) Nota AI GmbH. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn

from mmengine.model.weight_init import trunc_normal_
from mmpretrain.registry import MODELS
from .cls_head import ClsHead

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .cls_head import ClsHead


@MODELS.register_module()
class EfficientViTClsHead(ClsHead):
    def __init__(self, in_channels, num_classes, bias=True, std=0.02, drop=0.,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(EfficientViTClsHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)

        self.bn = nn.BatchNorm1d(in_channels)
        self.drop = nn.Dropout(drop)
        self.linear = nn.Linear(in_channels, num_classes, bias=bias)

        trunc_normal_(self.linear.weight, std=std)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, linear = self.bn, self.linear
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = linear.weight * w[None, :]
        if linear.bias is None:
            b = b @ self.linear.weight.T
        else:
            b = (linear.weight @ b[:, None]).view(-1) + self.linear.bias
        m = nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
    
    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.linear(self.drop(self.bn(pre_logits)))

        return cls_score

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In :obj`EfficientFormerClsHead`, we just
        obtain the feature of the last stage.
        """
        # The EfficientFormerClsHead doesn't have other module, just return
        # after unpacking.
        return feats[-1]