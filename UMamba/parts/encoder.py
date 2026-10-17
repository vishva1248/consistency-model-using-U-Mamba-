import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as f
from .noise_condition import NoiseCondition
from .double_conv import double_conv
from .mamba_block import MambaBlock
def encoder(x,t,device,save_tools=None):
    x = x.float()
    pool = nn.MaxPool2d(kernel_size=2,stride=2)
    skip_connection = []
    if save_tools is None:
        save_tools = nn.ModuleList()
        current_in_channel = x.shape[1]
        current_out_channel = 64 
        while x.shape[2] > 8:
            noise_layer = NoiseCondition(current_out_channel)
            layer = nn.ModuleDict({
                "conv": double_conv(current_in_channel, current_out_channel),
                "noise": NoiseCondition(current_out_channel),
                "mamba": MambaBlock(current_out_channel),
            }).to(x.device)
            x = layer["conv"](x)
            x = layer["noise"](x, t)
            x = layer["mamba"](x)
            skip_connection.append(x)
            x = pool(x)
            current_in_channel = current_out_channel
            current_out_channel = current_out_channel * 2
            save_tools.append(layer)
    else:
        for layer in save_tools:
            x = layer["conv"](x)
            x = layer["noise"](x, t)
            x = layer["mamba"](x)
            skip_connection.append(x)
            x = pool(x)
    return x,skip_connection,save_tools