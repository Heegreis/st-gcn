import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.graph import Graph

from net.utils.st_gcn_encoder import Encoder
from net.utils.st_gcn_decoder import Decoder

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            edge_importance = [1] * len(self.st_gcn_networks)

        self.encoder = Encoder(in_channels, kernel_size, A, edge_importance, **kwargs)
        self.decoder = Decoder(in_channels, kernel_size, A, edge_importance, **kwargs)
        
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x