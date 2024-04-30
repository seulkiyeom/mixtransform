# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from pathlib import Path

import torch
import timm

from mmpretrain.models.backbones import EfficientViT
from mmengine.runner import load_checkpoint, save_checkpoint


def convert_weights(weight):
    """Weight Converter.

    Converts the weights from timm to mmpretrain
    Args:
        weight (dict): weight dict from timm
    Returns:
        Converted weight dict for mmpretrain
    """
    result = dict()
    result['state_dict'] = weight.state_dict()


    # mapping = {
    #     'dwconv': 'depthwise_conv',
    #     'pwconv1': 'pointwise_conv1',
    #     'pwconv2': 'pointwise_conv2',
    #     'xca': 'csa',
    #     'convs': 'conv_modules',
    #     'token_projection': 'proj',
    #     'pos_embd': 'pos_embed',
    #     'temperature': 'scale',
    # }
    # strict_mapping = {
    #     'norm.weight': 'norm3.weight',
    #     'norm.bias': 'norm3.bias',
    # }

    # try:
    #     weight = weight['model']
    # except:
    #     raise NotImplementedError

    # for k, v in weight.items():
    #     # keyword mapping
    #     for mk, mv in mapping.items():
    #         if mk in k:
    #             k = k.replace(mk, mv)
    #     # strict mapping
    #     for mk, mv in strict_mapping.items():
    #         if mk == k:
    #             k = mv

    #     if k.startswith('head.'):
    #         temp['head.fc.' + k[5:]] = v
    #     else:
    #         temp['backbone.' + k] = v

    # result['state_dict'] = temp
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('arch', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    dst = Path(args.dst)
    if dst.suffix != '.pth':
        print('The path should contain the name of the pth format file.')
        exit(1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    original_model = timm.create_model(model_name=args.arch, pretrained=True, num_classes=1000, in_chans=3)
    converted_model = convert_weights(original_model)

    save_checkpoint(converted_model, args.dst)
    print(f'Saved Done')
    # backbone = EfficientViT(arch = 'm0')
    # # imgs = torch.randn(1, 3, 224, 224)
    # # feat = mymodel(imgs)
    # # print(feat.shape)
    # load_checkpoint(backbone, args.dst, strict=True)
