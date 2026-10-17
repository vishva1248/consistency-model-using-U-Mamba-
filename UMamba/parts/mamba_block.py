import torch
import torch.nn as nn
from .selective_ssm import SelectiveSSM
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        d_inner = 2 * d_model
        self.in_proj = nn.Linear(d_model,2 * d_inner)
        self.conv = nn.Conv1d(in_channels=d_inner,out_channels=d_inner,kernel_size=3,padding=1,groups=d_inner)
        self.silu = nn.SiLU()
        self.ssm = SelectiveSSM(d_state,d_inner)
        self.o_proj= nn.Linear(d_inner,d_model)
    def forward(self, x):
        B,C,H,W = x.shape
        residual = x
        x = x.permute(0,2,3,1)
        x = x.reshape(B,H*W,C)
        x = self.in_proj(x)
        x_main, x_gate = torch.chunk(x,2,dim=-1)
        x_main = x_main.transpose(1, 2)
        x_main = self.conv(x_main)
        x_main = x_main.transpose(1,2)
        x_main = self.silu(x_main)
        x_main_SSM = self.ssm(x_main)
        x_gate = self.silu(x_gate)
        mul_output = x_main_SSM * x_gate
        mamba_block_output = self.o_proj(mul_output)
        mamba_block_output = mamba_block_output.reshape(B,H,W,C)
        mamba_block_output = mamba_block_output.permute(0,3,1,2)
        mamba_block_output = mamba_block_output + residual
        return mamba_block_output
#completed