import torch
import torch.nn as nn
class SelectiveSSM(nn.Module):
    def __init__(self, d_state,d_inner):
        super().__init__()
        self.d_state = d_state
        self.d_inner = d_inner
        self.d_proj = nn.Linear(d_inner, d_inner)
        self.softplus = nn.Softplus()
        self.B_proj = nn.Linear(d_inner,d_state)
        self.C_proj = nn.Linear(d_inner,d_state)
        self.A = nn.Parameter(torch.rand(d_inner, d_state) * -1.0)
    def forward(self, x):
        delta = self.d_proj(x)
        delta = self.softplus(delta)
        B = self.B_proj(x)
        C = self.C_proj(x)
        delta_expanded = delta.unsqueeze(-1)
        A_bar = torch.exp(delta_expanded*self.A)
        B_expanded = B.unsqueeze(2)
        B_bar = delta_expanded*B_expanded
        batch_size , seq_len , _ = x.shape
        h = torch.zeros(batch_size,self.d_inner,self.d_state,device=x.device)
        states=[]
        for t in range(seq_len):
            x_t = x[:,t,:].unsqueeze(-1)
            A_bar_t = A_bar[:,t,:,:]
            b_bar_t = B_bar[:,t,:,:]
            h = A_bar_t * h + b_bar_t * x_t
            states.append(h)
        states= torch.stack(states,dim = 1)
        c_expanded = C.unsqueeze(2)
        output = c_expanded * states
        output = output.sum(dim=-1)
        return output
#completed