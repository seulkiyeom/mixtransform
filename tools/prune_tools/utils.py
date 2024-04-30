import torch
import torch.nn as nn
import numpy as np
from prune_tools.arch_modif import prune_layer
from mmengine.model import ModuleList, Sequential

def get_flops(module_type, info, lamb_in=None, lamb_out=None, train=True):
    """
    Remark: these functions are adapted to match the thop library
    """
    cin = info['cin'] if lamb_in is None else sum([torch.sum(l) for l in lamb_in])
    cout = info['cout'] if lamb_out is None else sum([torch.sum(l) for l in lamb_out])

    if not train:
        if cin is not None: cin = float(cin)
        if cout is not None: cout = float(cout)
    # print()
    # print(cin)
    # print(cout)
    # print()

    assert(cin<=info['cin'] and cout<=info['cout'])
    hw = info['h'] * info['w']
    return get_flops_conv2d(info, cin, cout, hw) + get_flops_batchnorm2d(info, cin, cout, hw) #Conv2d + BN
    # if module_type is nn.Conv2d:
    #     return get_flops_conv2d(info, cin, cout, hw)
    # elif module_type is nn.Linear:
    #     return get_flops_linear(info, cin, cout, hw)
    # elif module_type is nn.BatchNorm2d:
    #     return get_flops_batchnorm2d(info, cin, cout, hw)
    # elif module_type is nn.ReLU:
    #     return get_flops_ReLU(info, cin, cout, hw)
    # elif module_type is nn.AdaptiveAvgPool2d:
    #     return get_flops_AdaptiveAvgPool2d(info, cin, cout, hw)
    # elif module_type is nn.AvgPool2d:
    #     return get_flops_AvgPool2d(info, cin, cout, hw)
    # else: return 0

def get_flops_conv2d(info, cin, cout, hw):
    total_flops = 0
    kk = info['k']**2 
    # bias = Cout x H x W if model has bias, else 0
    bias_flops = 0 #cout*hw if info['has_bias'] else 0
    # Cout x Cin x H x W x Kw x Kh
    return cout * cin * hw * kk + bias_flops

def get_flops_linear(info, cin, cout, hw):
    #bias_flops = cout if info['has_bias'] else 0
    return cin * cout #* hw + bias_flops

def get_flops_batchnorm2d(info, cin, cout, hw):
    return hw * cin * 2

def get_flops_ReLU(info, cin, cout, hw):
    return 0 #hw * cin

def get_flops_AdaptiveAvgPool2d(info, cin, cout, hw):
    kk = info['k']**2
    return (kk+1) * hw * cout

def get_flops_AvgPool2d(info, cin, cout, hw):
    return hw * cout

def set_in_index_attr(model, module_name, value):
    # 모듈 이름을 '.' 기준으로 나누어 리스트 생성
    path = module_name.split('.')
    
    # 초기 모듈을 model로 설정
    module = model
    
    # path를 순회하며 각 단계에서 모듈을 업데이트
    for p in path:
        if p.isdigit():  # 숫자면 인덱스로 변환
            module = module[int(p)]
        else:  # 숫자가 아니면 속성으로 접근
            module = getattr(module, p)
    
    # 최종 모듈에 'in_index' 속성 설정
    setattr(module, 'in_index', torch.tensor(value))

def set_in_index_attr_test(model, module_name, value):
    # 모듈 이름을 '.' 기준으로 나누어 리스트 생성
    path = module_name.split('.')
    
    # 초기 모듈을 model로 설정
    module = model
    
    # path를 순회하며 각 단계에서 모듈을 업데이트
    for p in path:
        if p.isdigit():  # 숫자면 인덱스로 변환
            module = module[int(p)]
        else:  # 숫자가 아니면 속성으로 접근
            module = getattr(module, p)
    
    # 최종 모듈에 'in_index' 속성 설정
    setattr(module, 'in_index', value)

def pruning(model, preserved_indexes_all):
    # |Good parts:
    # |1. The code iterates through the named modules in the model, allowing for targeted pruning of specific layers based on their names.
    # |2. It keeps track of the indexes to be pruned using `index_stack` and updates the count and value number accordingly.
    # |3. It prints out information about the pruning target and the pruning ratio for each layer being pruned.
    # |
    # |Bad parts:
    # |1. The code lacks comments to explain the purpose of each section and the overall logic behind the pruning process.
    # |2. The variable names could be more descriptive to improve readability and maintainability.
    # |3. The logic for determining when to prune DWConv and FFN layers could be extracted into separate functions for better modularity.
    # |4. The code could beknefit from error handling in case the expected modules are not found in the model.
    # |5. The use of global variables like `prune_FFN` could lead to potential issues with code maintainability and readability. Consider passing necessary variables as arguments instead.
    previous_pruned = False
    tot_layer = len(preserved_indexes_all)
    cnt = 0
    value_num = 0
    index_stack = []

    for name_module, module in model.named_modules():
        if 'mixer.m.attn.mix' in name_module and not isinstance(module, ModuleList):
            index_stack += [item + value_num*module.in_channels for item in preserved_indexes_all[cnt]]
            prune_ratio = prune_layer(model, name_module, preserved_indexes_all[cnt], both_prune=True)
            print(f'[{cnt}] Target: {name_module} ({int(prune_ratio*100)}%)')            
            set_in_index_attr(model, name_module, preserved_indexes_all[cnt])
            prune_FFN = True
            cnt += 1
            value_num += 1

        elif 'proj' in name_module and isinstance(module, nn.Conv2d) and prune_FFN is True:
            prune_ratio = prune_layer(model, name_module, index_stack, both_prune=False)
            print(f'[{cnt}] Target: {name_module} ({int(prune_ratio*100)}%)')
            prune_FFN = False
            value_num = 0
            index_stack = []

def prune_model_align(model, checkpoint):
    index_stack = []
    for name_module, module in model.named_modules():
        if 'mixer.m.attn.mix' in name_module and not isinstance(module, nn.ModuleList):
            index_stack += range(checkpoint[name_module + '.weight'].size(0))
            prune_layer(model, name_module, range(checkpoint[name_module + '.weight'].size(0)), both_prune=True)
            prune_FFN = True
            # checkpoint[name_module + '.weight'].size(0)

        elif 'proj' in name_module and isinstance(module, nn.Conv2d) and prune_FFN is True:
            prune_layer(model, name_module, range(len(index_stack)), both_prune=False)
            prune_FFN = False
            index_stack = []
    return model
