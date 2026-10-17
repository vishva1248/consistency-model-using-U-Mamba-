import torch.nn as nn
def double_conv(in_channel,out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel,out_channel,kernel_size= 3 , padding = 1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel,out_channel,kernel_size = 3 , padding = 1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )