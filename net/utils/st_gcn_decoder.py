import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.graph import Graph
from net.utils.de_st_gcn import de_st_gcn

class Decoder(nn.Module):
    def __init__(self, in_channels, kernel_size,
                 edge_importance_weighting, graph_args, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.st_gcn_networks = nn.ModuleList((
            de_st_gcn(256, 256, kernel_size, 1, **kwargs),
            de_st_gcn(256, 256, kernel_size, 1, **kwargs),
            de_st_gcn(256, 128, kernel_size, 2, **kwargs),
            de_st_gcn(128, 128, kernel_size, 1, **kwargs),
            de_st_gcn(128, 128, kernel_size, 1, **kwargs),
            de_st_gcn(128, 64, kernel_size, 2, **kwargs),
            de_st_gcn(64, 64, kernel_size, 1, **kwargs),
            de_st_gcn(64, 64, kernel_size, 1, **kwargs),
            de_st_gcn(64, 64, kernel_size, 1, **kwargs),
            de_st_gcn(64, in_channels, kernel_size, 1, residual=False, **kwargs0),
        ))
    
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            edge_importance = [1] * len(self.st_gcn_networks)

        self.edge_importance = edge_importance

    def forward(self, x):
        for de_gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = de_gcn(x, self.A * importance)
        
        return x