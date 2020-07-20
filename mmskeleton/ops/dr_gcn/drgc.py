import torch
import torch.nn as nn


class SpatialDyadicRelationalGC(nn.Module):
    r"""Dyadic Relational Graph Convolution and Spatial Graph Convolution.
        For simplicity, we wrote them together.
        Input graph is divided into two subgraphs, physical graph
            and relational graphs. Physical graph is presented by a
            static adjacency matrix. Relational graphs are presented by RAM.
            Each relational graph is constructed for one corresponding frame.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the SGC kernel
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:
            `(N, M, in_channels, T_{in}, V)` format
        - Input[1]: Relative adjacency matrix (RAM)
            in :math:`(N, T_{int}, V, V)` format
        - Input[2]: Input physical graph adjacency matrix
            in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence
            in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: RAM after temporal convolution
            in :math:`(N, T_{out}, V, V)` format
        - Output[2]: Physical graph adjacency matrix for output data
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

        self.conv_RAM = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=(1, 1),
                              bias=bias)
        self.conv = nn.Conv2d(in_channels,
                              out_channels*kernel_size,
                              kernel_size=(1, 1),
                              bias=bias)

    def forward(self, x, RAM, A):
        # split input data to two actors
        x_person_a, x_person_b = x.chunk(2,1)
        x_person_a = torch.squeeze(x_person_a,1)
        x_person_b = torch.squeeze(x_person_b,1)

        # SGC begin
        # multiply weights and do channel transformation for SGC
        x_person_a_s = self.conv(x_person_a)
        x_person_b_s = self.conv(x_person_b)
        n, kc, t, v = x_person_a_s.size()
        x_person_a_s = x_person_a_s.view(
            n, self.kernel_size, kc // self.kernel_size, t, v)
        x_person_b_s = x_person_b_s.view(
            n, self.kernel_size, kc // self.kernel_size, t, v)
        x_person_a_s = torch.einsum('nkctv,kvw->nctw', (x_person_a_s, A))
        x_person_b_s = torch.einsum('nkctv,kvw->nctw', (x_person_b_s, A))
        # SGC done

        # DRGC begin
        # mupltiply weights and do channel transformation for DRGC
        x_person_a = self.conv_RAM(x_person_a)
        x_person_b = self.conv_RAM(x_person_b)
        x_person_a_dr = torch.einsum('nctw,ntvw->nctv', (x_person_b, RAM))
        x_person_b_dr = torch.einsum('nctv,ntvw->nctw', (x_person_a, RAM))
        # DRGC done

        # Add them together
        x_person_a = x_person_a_s+x_person_a_dr
        x_person_b = x_person_b_s+x_person_b_dr

        # stack two people's data
        x = torch.stack((x_person_a,x_person_b), 1)

        return x.contiguous(), RAM, A
