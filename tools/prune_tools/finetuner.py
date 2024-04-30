import torch
import torch.nn as nn
from prune_tools.bottleneck import ModuleInfo, ModulesInfo

def get_modules(model, device):
    module_list = []
    init_lamb = []
    for name, module in model.named_modules():
        if 'mix' in name and isinstance(module, nn.ModuleList): #select values that ara target for pruning only
            for i in range(len(module)): #head 개수만큼 load
                module_list.append(ModuleInfo(module[i]))
                init_lamb.append(torch.tensor([0.9] * module[i].weight.size(0), dtype=torch.float32))
                # init_lamb.append(torch.tensor([0.9] * module[i].c.weight.size(0), dtype=torch.float32)) #Cream에서 제공하는 Conv + BN 쓰고 싶을 경우
    modules = ModulesInfo(model, module_list, input_img_size=224, device=device)
    return modules, init_lamb