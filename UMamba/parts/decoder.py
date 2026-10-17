import torch
import torch.nn as nn
from .double_conv import double_conv
def decoder(x,skip_connection,save_tools=None):
    if save_tools is None:
        save_tools = nn.ModuleDict({'up': nn.ModuleList(),'blend': nn.ModuleList()})
        for skip in reversed(skip_connection):
            current_in_channel = x.shape[1]
            current_out_channel = skip.shape[1]
            up_layer = nn.ConvTranspose2d(current_in_channel,current_out_channel,kernel_size=2,stride=2).to(x.device)
            blend_layer = double_conv(current_out_channel*2,current_out_channel).to(x.device)
            save_tools['up'].append(up_layer)
            save_tools['blend'].append(blend_layer)
            #decoder work start here
            up_x = up_layer(x)
            concat_x = torch.cat((skip,up_x),dim=1)
            x = blend_layer(concat_x)
        final_layer = nn.Conv2d(x.shape[1],3,kernel_size=1).to(x.device)
        save_tools['final'] = final_layer
        prediction  = final_layer(x)
    else:
        for i ,skip in enumerate(reversed(skip_connection)):
            up_layer = save_tools['up'][i]
            blend_layer = save_tools['blend'][i]
            up_x = up_layer(x)
            concat_x = torch.cat((skip,up_x),dim=1)
            x = blend_layer(concat_x)
        prediction = save_tools['final'](x)
    return prediction , save_tools