import torch.nn as nn
from custom_op.conv2d.conv_avg import Conv2dAvg
from custom_op.conv2d.conv_svd import Conv2d_SVD
from custom_op.conv2d.conv_hosvd import Conv2d_HOSVD

from custom_op.linear.linear_hosvd import Linear_HOSVD
from custom_op.linear.linear_svd import Linear_SVD
import torch

def get_all_linear_with_name(model):
    linear_layers = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.modules.linear.Linear) or isinstance(mod, Linear_HOSVD) or isinstance(mod, Linear_SVD):
            linear_layers[name] = mod
    return linear_layers

def get_active_linear_with_name(model):
    total_linear_layer = get_all_linear_with_name(model)
    if model.num_of_finetune == "all" or model.num_of_finetune > len(total_linear_layer):
        return total_linear_layer
    elif model.num_of_finetune == None or model.num_of_finetune == 0:
        return -1
    else:
        active_linear_layers = dict(list(total_linear_layer.items())[-model.num_of_finetune:])
        return active_linear_layers

def get_all_conv_with_name(model):
    conv_layers = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.modules.conv.Conv2d) or isinstance(mod, Conv2dAvg) or isinstance(mod, Conv2d_SVD) or isinstance(mod, Conv2d_HOSVD):
            conv_layers[name] = mod
    return conv_layers
    
def get_active_conv_with_name(model):
    total_conv_layer = get_all_conv_with_name(model)
    if model.num_of_finetune == "all" or model.num_of_finetune > len(total_conv_layer):
        return total_conv_layer
    elif model.num_of_finetune == None or model.num_of_finetune == 0:
        return -1
    else:
        active_conv_layers = dict(list(total_conv_layer.items())[-model.num_of_finetune:])
        return active_conv_layers

class Hook:
    def __init__(self, module):
        self.module = module
        self.input_size = torch.zeros(4)
        self.output_size = torch.zeros(4)
        
        self.inputs = []#torch.empty(0, 4)
        
        self.active = True
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        if not self.active:
            return
        Input = input[0].clone().detach()
        Output = output.clone().detach()
        self.input_size = Input.shape
        self.output_size = Output.shape

        self.inputs.append(Input)
    def activate(self, active):
        self.active = active
    def remove(self):
        self.active = False
        self.hook.remove()
        # print("Hook is removed")

def attach_hooks_for_conv(model, consider_active_only=False):
    if not consider_active_only:
        conv_layers = get_all_conv_with_name(model)
    else:
        conv_layers = get_active_conv_with_name(model)
    assert conv_layers != -1, "[Warning] Consider activate conv2d only but no conv2d is finetuned => No hook is attached !!"

    for name, mod in  conv_layers.items():
        model.hook[name] = Hook(mod)

def attach_hooks_for_linear(model, consider_active_only=False):
    if not consider_active_only:
        linear_layers = get_all_linear_with_name(model)
    else:
        linear_layers = get_active_linear_with_name(model)
    assert linear_layers != -1, "[Warning] Consider activate conv2d only but no conv2d is finetuned => No hook is attached !!"

    for name, mod in  linear_layers.items():
        model.hook[name] = Hook(mod)