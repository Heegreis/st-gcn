import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.graph import Graph

# from net.utils.st_gcn_encoder import Encoder
# from net.utils.st_gcn_decoder import Decoder

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x