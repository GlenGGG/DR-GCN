import torch
import torch.nn as nn
from mmskeleton.ops.dr_gcn import SpatialGC


class RAMGen(nn.Module):
    r"""Relative Attention Matrix (RAM) generation process.

    Args:
        in_channels (int): Number of channels in the input sequence data
        encoder_out_channels (int): Number of output channels of encoder
        decoder_out_channels (int): Number of output channels of decoder
        kernel_size (tuple):
            Size of the temporal kernel and SGC kernel
        T (int): Number of frames in input sequence data.

    Shape:
        - Input[0]: Input graph sequence in :math:
            `(N, M, in_channels, T, V)` format
        - Input[1]: Input physical graph adjacency matrix
            in :math:`(K, V, V)` format
        - Output[0]: RAM in :math:`(N, T, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`M` is the number of instance in a frame.
            :math:`K` is the spatial kernel size,
                as :math:`K == kernel_size[1]`,
            :math:`T` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self, in_channels,
                 encoder_out_channels,
                 decoder_out_channels,
                 kernel_size,
                 T,
                 relative_attention_component=True,
                 geometric_component=True,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.relative_attention_component = relative_attention_component
        self.geometric_component = geometric_component
        if relative_attention_component and geometric_component:
            self.integration_param = nn.Parameter(
                torch.FloatTensor([0.0]), requires_grad=True)

        # batchnorm before rectification
        self.RAM_batchnorm2d = nn.BatchNorm2d(T)

        # batchnorm after encoder
        self.encoder_batchnorm2d = nn.BatchNorm2d(encoder_out_channels)
        # batchnorm after decoder
        self.decoder_batchnorm2d_out = nn.BatchNorm2d(
            decoder_out_channels)

        if self.relative_attention_component:
            self.encoder = SpatialGC(
                in_channels, encoder_out_channels, kernel_size[1])
            self.decoder = SpatialGC(
                encoder_out_channels,
                decoder_out_channels,
                kernel_size[1])

    def forward(self, x, A):

        N,M,C,T,V=x.shape
        x_person_a, x_person_b = x.chunk(2, 1)
        x_person_a = torch.squeeze(x_person_a, 1)
        x_person_b = torch.squeeze(x_person_b, 1)
        x = []

        if self.geometric_component:
            RAM_g_p1 = torch.exp(x_person_a)
            RAM_g_p2 = torch.exp(-x_person_b)
            RAM_g = torch.einsum(
                'nctv, nctw->nctvw', RAM_g_p1, RAM_g_p2)
            RAM_g = (torch.log(RAM_g+1e-5)).pow(2)
            RAM_g = (-1/self.in_channels)*(
                torch.einsum('nctvw->ntvw', RAM_g))
            RAM_g = torch.exp(RAM_g)
        if self.relative_attention_component:
            # encoder begin
            RAM_r_p1, _ = self.encoder(x_person_a, A)
            RAM_r_p2, _ = self.encoder(x_person_b, A)
            RAM_r_p1 = self.encoder_batchnorm2d(RAM_r_p1)
            RAM_r_p2 = self.encoder_batchnorm2d(RAM_r_p2)
            RAM_r_p1 = torch.relu(RAM_r_p1)
            RAM_r_p2 = torch.relu(RAM_r_p2)
            # encoder done
            # decoder begin
            RAM_r_p1, _ = self.decoder(RAM_r_p1, A)
            RAM_r_p2, _ = self.decoder(RAM_r_p2, A)
            RAM_r_p1 = self.decoder_batchnorm2d_out(RAM_r_p1)
            RAM_r_p2 = self.decoder_batchnorm2d_out(RAM_r_p2)
            # decoder done
            # calculate cosine distance
            RAM_r = torch.einsum(
                'nctv, nctw->ntvw',
                RAM_r_p1,
                RAM_r_p2)
            RAM_r = self.RAM_batchnorm2d(RAM_r)
            # map to [0,1]
            RAM_r = scaledTanh(RAM_r)

        if self.relative_attention_component and self.geometric_component:
            RAM_gen_percentile = scaledTanh(self.integration_param)
            RAM = RAM_gen_percentile * RAM_r + (
                1 - RAM_gen_percentile) * RAM_g
        elif self.relative_attention_component:
            RAM = RAM_r
        elif self.geometric_component:
            RAM = RAM_g
        else:
            RAM = torch.ones(N,T,V,V)

        abnormal = torch.sum(torch.var(x_person_b, dim=3), dim=1)
        abnormal = torch.where(
            abnormal < 1e-4,
            torch.zeros_like(abnormal),
            torch.ones_like(abnormal))
        RAM = torch.einsum('ntvw,nt->ntvw', RAM, abnormal)
        RAM = normalizeRAM(RAM)

        return RAM


def normalizeRAM(RAM):

    RAM = torch.where(RAM < 0.5, torch.zeros_like(RAM).cuda(), RAM)
    da = torch.sum(RAM, dim=2)  # sum over column
    db = torch.sum(RAM, dim=3)  # sum over line
    norm = torch.einsum('ntv,ntw->ntvw', db, da)
    norm = torch.pow((norm+1e-5), -0.5)
    RAM = RAM*norm

    return RAM


def scaledTanh(param):

    return (torch.tanh(param)+1.0)/2.0
