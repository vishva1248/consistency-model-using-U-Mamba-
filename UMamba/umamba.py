import torch
import torch.nn as nn
from .parts.encoder import encoder
from .parts.decoder import decoder
# print("hello")
class Umamba(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.encoder_tools = None
        self.decoder_tools = None
    def forward(self,x,t):
        if t.ndim == 0:
            t = t.expand(x.shape[0])

        t = t.to(device=x.device, dtype=x.dtype).reshape(-1)

        if t.numel() == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])

        if t.shape[0] != x.shape[0]:
            raise ValueError(
                f"Expected one noise level per image; "
                f"got {t.shape[0]} for batch {x.shape[0]}"
            )
        sigma_data = self.sigma_data
        t_min = self.sigma_min
        c_in = torch.rsqrt(t.square() + sigma_data ** 2)
        c_skip = (sigma_data ** 2/ ((t - t_min).square() + sigma_data ** 2))
        c_out = (sigma_data * (t - t_min)/ torch.sqrt(t.square() + sigma_data ** 2))
        shape = (-1, 1, 1, 1)
        model_input = c_in.view(shape) * x
        encode_image, skip, save_tools = encoder(model_input, t, self.device, self.encoder_tools)
        self.encoder_tools = save_tools
        output, save_tools = decoder(encode_image, skip, self.decoder_tools)
        self.decoder_tools = save_tools
        output = (
                c_skip.view(shape) * x
                + c_out.view(shape) * output
        )
        return output