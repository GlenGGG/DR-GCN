import torch
import torch.nn as nn


class SpatialGC(nn.Module):
    r"""Sapatial Graph Convolution used in DR-GCB and RAM_r's
            encoder and decoder

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence
            in :math:`(N, M, in_channels, T_{in}, V)` format
        - Input[1]: Input physical graph adjacency matrix
            in :math:`(K, V, V)` format
        - Output[0]: Output physical graph sequence
            in :math:`(N, M, out_channels, T_{out}, V)` format
        - Output[1]: Physical graph adjacency matrix for output data
            in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`M` is the number of instance in a frame.
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels*kernel_size,
                              kernel_size=(1, 1),
                              bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A
