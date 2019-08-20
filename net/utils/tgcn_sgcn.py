# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

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
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A, importance):
        A_list = []
        for i in range(0, importance.size()[0]):
            A_list.append(A * importance[i, :, :, :])
        A = torch.stack(A_list, dim=0)

        n, c, t, v = x.size()
        o, k, v, w = A.size()

        yo_list = []

        for index_o in range(0, o):
            Ao = A[index_o, :, :, :]
            for index_c in range(0, c):
                xc = x[:, index_c, :, :]  # n, t, v
                yc = torch.einsum('ntv,kvw->nktw', (xc, Ao))
                # sum k
                yck = yc[:, 0, :, :]
                for index_k in range(1, k):
                    yck =  yck.add(yc[:, index_k, :, :])
                yc = yck # n, t, w
                if index_c == 0:
                    yc_tmp = yc
                else:
                    yc_tmp.add(yc)
            yo_list.append(yc_tmp)
        y = torch.stack(yo_list, dim=1) # n, o, t, w = n, c, t, v

        return y, A
