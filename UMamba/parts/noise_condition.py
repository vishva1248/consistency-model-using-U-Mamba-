import math
import torch
import torch.nn as nn
class NoiseCondition(nn.Module):
    def __init__( self, channels, embedding_dim=256,sigma_min=0.002, sigma_max=80.0,):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even")
        if sigma_min <= 0 or sigma_max <= sigma_min:
            raise ValueError("Require 0 < sigma_min < sigma_max")
        self.embedding_dim = embedding_dim
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, channels),)
    def forward(self, x, sigma):
        sigma = sigma.to(device=x.device,dtype=x.dtype,).reshape(-1)
        if sigma.numel() == 1 and x.shape[0] > 1:
            sigma = sigma.expand(x.shape[0])
        if sigma.shape[0] != x.shape[0]:
            raise ValueError(f"Expected {x.shape[0]} noise levels, "f"received {sigma.shape[0]}")
        sigma = sigma.clamp(min=self.sigma_min,max=self.sigma_max,)
        log_sigma = torch.log(sigma)
        log_min = math.log(self.sigma_min)
        log_max = math.log(self.sigma_max)
        normalized_log_sigma = ((log_sigma - log_min) / (log_max - log_min))
        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half_dim,device=x.device,dtype=x.dtype,)/ max(half_dim - 1, 1))
        angles = (1000.0 * normalized_log_sigma.unsqueeze(1) * frequencies.unsqueeze(0))
        embedding = torch.cat([torch.sin(angles),torch.cos(angles),],dim=-1,)
        projected = self.projection(embedding)
        projected = projected.unsqueeze(-1).unsqueeze(-1)
        return x + projected