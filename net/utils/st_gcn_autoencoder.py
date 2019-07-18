import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.graph import Graph

# from net.utils.st_gcn_encoder import Encoder
# from net.utils.st_gcn_decoder import Decoder

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, data_bn):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.data_bn = data_bn
    
    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous() # N, M, V, C, T
        x = x.view(N * M, V * C, T) # N*M, V*C, T
        x = self.data_bn(x) # N*M, V*C, T
        x = x.view(N, M, V, C, T) # N, M, V, C, T
        x = x.permute(0, 1, 3, 4, 2).contiguous() # N, M, C, T, V
        x = x.view(N * M, C, T, V) # N*M, C, T, V

        x = self.encoder(x)
        x = self.decoder(x)

        # data normalization back # N*M, C, T, V
        x = x.view(N, M, C, T, V) # N, M, C, T, V
        x = x.permute(0, 1, 4, 2, 3).contiguous() # N, M, V, C, T
        x = x.view(N*M, V*C, T) # N*M, V*C, T
        x = self.data_bn(x) # N*M, V*C, T
        x = x.view(N, M, V, C, T) # N, M, V, C, T
        x = x.permute(0, 3, 4, 2, 1).contiguous() # N, C, T, V, M

        return x