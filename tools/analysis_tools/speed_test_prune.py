"""
Testing the speed of different models
"""
import os
import torch
import time
import re
from collections import OrderedDict
import torch.nn as nn
from tools.prune_tools.utils import set_in_index_attr_test
# from model.build import EfficientViT_M0, EfficientViT_M1, EfficientViT_M2, EfficientViT_M3, EfficientViT_M4, EfficientViT_M5
# from mmpretrain.models.backbones import EfficientViT
from mmpretrain.models.backbones import MixViT
# from tools.prune_tools.runner_prune import CustomRunner
# from mmengine.runner import Runner
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from mmengine.model import ModuleList
from tools.prune_tools.arch_modif import prune_layer

T0 = 10
T1 = 60

path = {
    'm0': None, #m0 pruned model
    'm1': "work_dirs/mixvit-m1_8xb32_in1k/20240429_162456/best_accuracy_top1_epoch_99.pth", #m1 pruned model
    'm2': None, #m2 pruned model
    'm3': None, #m3 pruned model
    'm4': None, #m0 pruned model
    'm5': None, #m0 pruned model
}


def compute_throughput_cpu(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    # warmup
    start = time.time()
    while time.time() - start < T0:
        model(inputs)

    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

def compute_throughput_cuda(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast():
        while time.time() - start < T0:
            model(inputs)
    timing = []
    if device == 'cuda:0':
        torch.cuda.synchronize()
    with torch.cuda.amp.autocast():
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)

def load_pruned_model(model, path):
    checkpoint = _load_checkpoint(path)['state_dict']
    index_stack = []
    for name_module, module in model.named_modules():
        if 'mixer.m.attn.mix' in name_module and not isinstance(module, ModuleList):
            index_stack += range(checkpoint['backbone.' + name_module + '.weight'].size(0))
            prune_layer(model, name_module, range(checkpoint['backbone.' + name_module + '.weight'].size(0)), both_prune=True)
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
        set_in_index_attr_test(model, k, v)

if __name__ == '__main__':
    torch.autograd.set_grad_enabled(False)

    for device in ['cuda:0', 'cpu']:

        if 'cuda' in device and not torch.cuda.is_available():
            print("no cuda")
            continue

        if device == 'cpu':
            os.system('echo -n "nb processors "; '
                    'cat /proc/cpuinfo | grep ^processor | wc -l; '
                    'cat /proc/cpuinfo | grep ^"model name" | tail -1')
            print('Using 1 cpu thread')
            torch.set_num_threads(1)
            compute_throughput = compute_throughput_cpu
        else:
            print(torch.cuda.get_device_name(torch.cuda.current_device()))
            compute_throughput = compute_throughput_cuda

        for n, batch_size0, resolution in [
            ('m0', 2048, 224),
            ('m1', 2048, 224),
            ('m2', 2048, 224),
            ('m3', 2048, 224),
            ('m4', 2048, 224),
            ('m5', 2048, 224),
        ]:

            if device == 'cpu':
                batch_size = 16
            else:
                batch_size = batch_size0
                torch.cuda.empty_cache()
            inputs = torch.randn(batch_size, 3, resolution,
                                resolution, device=device)
            model = MixViT(arch = n)
            load_pruned_model(model, path[n])

            # replace_batchnorm(model)
            model.to(device)
            model.eval()
            model = torch.jit.trace(model, inputs)
            compute_throughput(n, model, device,
                            batch_size, resolution=resolution)

