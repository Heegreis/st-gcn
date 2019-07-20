import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.graph import Graph

# from net.utils.st_gcn_encoder import Encoder
# from net.utils.st_gcn_decoder import Decoder
from torchvision.utils import make_grid

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, data_bn):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.data_bn = data_bn
    
    def forward(self, x, writer = None, current_iter = None):
        if writer != None and current_iter != None:
            if current_iter % 10 == 0:
                writer.add_image('input_img', make_grid(x[0, :, :, :, 0].detach().cpu().unsqueeze(dim=1), nrow=3, padding=20, normalize=True, pad_value=1), current_iter)

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous() # N, M, V, C, T
        x = x.view(N * M, V * C, T) # N*M, V*C, T
        x = self.data_bn(x) # N*M, V*C, T
        x = x.view(N, M, V, C, T) # N, M, V, C, T
        x = x.permute(0, 1, 3, 4, 2).contiguous() # N, M, C, T, V
        x = x.view(N * M, C, T, V) # N*M, C, T, V
        
        
        if writer != None and current_iter != None:
            if current_iter % 10 == 0:
                writer.add_image('normalized_input_img', make_grid(x[0].detach().cpu().unsqueeze(dim=1), nrow=3, padding=20, normalize=True, pad_value=1), current_iter)

        x = self.encoder(x)

        if writer != None and current_iter != None:
            if current_iter % 10 == 0:
                writer.add_image('code_img', make_grid(x[0].detach().cpu().unsqueeze(dim=1), nrow=16, padding=10, normalize=True, pad_value=1), current_iter)
        
        x = self.decoder(x)

        if writer != None and current_iter != None:
            if current_iter % 10 == 0:
                writer.add_image('decode_img', make_grid(x[0].detach().cpu().unsqueeze(dim=1), nrow=3, padding=20, normalize=True, pad_value=1), current_iter)

        # data normalization back # N*M, C, T, V
        x = x.view(N, M, C, T, V) # N, M, C, T, V
        x = x.permute(0, 1, 4, 2, 3).contiguous() # N, M, V, C, T
        x = x.view(N*M, V*C, T) # N*M, V*C, T
        x = self.data_bn(x) # N*M, V*C, T
        x = x.view(N, M, V, C, T) # N, M, V, C, T
        x = x.permute(0, 3, 4, 2, 1).contiguous() # N, C, T, V, M

        if writer != None and current_iter != None:
            if current_iter % 10 == 0:
                writer.add_image('output_img', make_grid(x[0, :, :, :, 0].detach().cpu().unsqueeze(dim=1), nrow=3, padding=20, normalize=True, pad_value=1), current_iter)

        return x