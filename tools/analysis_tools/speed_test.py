"""
Testing the speed of different models
"""
import os
import torch
import torchvision
import time
import timm
# from model.build import EfficientViT_M0, EfficientViT_M1, EfficientViT_M2, EfficientViT_M3, EfficientViT_M4, EfficientViT_M5
import torchvision
import utils
from mmpretrain.models.backbones import MixViT
from mmpretrain.models.backbones import MlpMixer, RepVGG

T0 = 10
T1 = 60

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
            # ('m1', 2048, 224),
            # ('m2', 2048, 224),
            # ('m3', 2048, 224), #MixViT_M
            # ('m4', 2048, 224),
            # ('m5', 2048, 224), #MixViT_L
        ]:

            if device == 'cpu':
                batch_size = 16
            else:
                batch_size = batch_size0
                torch.cuda.empty_cache()
            inputs = torch.randn(batch_size, 3, resolution,
                                resolution, device=device)
            # model = MixViT(arch = n)
            # model = EfficientViT(arch = n)
            model = RepVGG(arch = 'A2') #A0, A1, B0, A2
            model.switch_to_deploy()
            # replace_batchnorm(model)
            model.to(device)
            model.eval()
            model = torch.jit.trace(model, inputs)
            compute_throughput(n, model, device,
                            batch_size, resolution=resolution)

