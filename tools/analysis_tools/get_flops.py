# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmengine.analysis import get_model_complexity_info
from mmpretrain import get_model
from mmengine.runner.checkpoint import _load_checkpoint
from mmengine.model import ModuleList
from tools.prune_tools.arch_modif import prune_layer
import torch.nn as nn
import re
from collections import OrderedDict
from tools.prune_tools.utils import set_in_index_attr_test

def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args

def load_pruned_model(model, path):
    checkpoint = _load_checkpoint(path)['state_dict']
    index_stack = []
    for name_module, module in model.named_modules():
        if 'mixer.m.attn.mix' in name_module and not isinstance(module, ModuleList):
            index_stack += range(checkpoint[name_module + '.weight'].size(0))
            prune_layer(model, name_module, range(checkpoint[name_module + '.weight'].size(0)), both_prune=True)
            prune_FFN = True

        elif 'proj' in name_module and isinstance(module, nn.Conv2d) and prune_FFN is True:
            prune_layer(model, name_module, range(len(index_stack)), both_prune=False)
            prune_FFN = False
            index_stack = []

    unpruned_indices = _load_checkpoint(path)['unpruned_indices'].copy()
    revise_keys: list = [(r'^backbone.', '')]
    for p, r in revise_keys:
        unpruned_indices = OrderedDict(
        {re.sub(p, r, k): v
            for k, v in unpruned_indices.items()})
        
    for k, v in unpruned_indices.items():       
        set_in_index_attr_test(model.backbone, k, v)

def main():
    args = parse_args()
    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    model = get_model(args.config)
    if 'prune' in args.checkpoint:
        load_pruned_model(model, args.checkpoint)
    model.eval()

    if hasattr(model, 'extract_feat'):
        model.forward = model.extract_feat
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
    analysis_results = get_model_complexity_info(
        model,
        input_shape,
    )
    flops = analysis_results['flops_str']
    params = analysis_results['params_str']
    activations = analysis_results['activations_str']
    out_table = analysis_results['out_table']
    out_arch = analysis_results['out_arch']
    print(out_arch)
    print(out_table)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n'
          f'Activation: {activations}\n{split_line}')
    print('!!!Only the backbone network is counted in FLOPs analysis.')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
