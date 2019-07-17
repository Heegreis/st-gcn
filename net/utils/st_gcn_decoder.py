import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.st_gcn import de_st_gcn

class Decoder(nn.Module):
    def __init__(self, in_channels, kernel_size,
                 A, edge_importance, **kwargs):
        super().__init__()

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.st_gcn_networks = nn.ModuleList((
            de_st_gcn(256, 256, kernel_size, 1, **kwargs),
            de_st_gcn(256, 256, kernel_size, 1, **kwargs),
            de_st_gcn(128, 256, kernel_size, 2, **kwargs),
            de_st_gcn(128, 128, kernel_size, 1, **kwargs),
            de_st_gcn(128, 128, kernel_size, 1, **kwargs),
            de_st_gcn(64, 128, kernel_size, 2, **kwargs),
            de_st_gcn(64, 64, kernel_size, 1, **kwargs),
            de_st_gcn(64, 64, kernel_size, 1, **kwargs),
            de_st_gcn(64, 64, kernel_size, 1, **kwargs),
            de_st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
        ))
    
        self.edge_importance = edge_importance
        self.A = A

    def forward(self, x):
        for de_gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = de_gcn(x, self.A * importance)
        
        return x