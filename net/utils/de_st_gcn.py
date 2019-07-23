import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.de_tgcn import DeConvTemporalGraphical

class de_st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = DeConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.de_tcn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            Interpolate(scale_factor=(1/stride, 1), mode='bilinear'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        x = self.de_tcn(x)
        x, A = self.gcn(x, A)

        return self.relu(x), A

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x