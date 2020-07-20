import torch
import torch.nn as nn
import torch.nn.functional as F

from mmskeleton.ops.dr_gcn import SpatialDyadicRelationalGC, Graph

from .ram_gen_process import RAMGen, scaledTanh, normalizeRAM


class DyadicRelationalGCN(nn.Module):
    r"""Dyadic Relational Graph olutional Network

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        T (int): Number of frames in input sequence data.
        RAM_encoder_output_channels (int): Output channels of the encoder in
            RAM_r generation.
        RAM_decoder_output_channels (int): Output channels of the decoder in
            RAM_r generation.
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        relative_attention_component (bool): If ``True``, add the relative
            attention component to RAM.
        geometric_component (bool): If ``True``, add the geometric comonent
            to RAM.
        temporal_kernel_size (int): Temporal Convolution kernel size.
        **kwargs (optional): Other parameters for graph olution units

    Shape:
        - Input: :math:`(N, M, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 T=300,
                 RAM_encoder_output_channels=128,
                 RAM_decoder_output_channels=64,
                 edge_importance_weighting=True,
                 relative_attention_component=True,
                 geometric_component=True,
                 temporal_kernel_size=9,
                 **kwargs):
        super().__init__()

        self.relative_attention_component = relative_attention_component
        self.geometric_component = geometric_component

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        self.temporal_kernel_size = temporal_kernel_size
        kernel_size = (
            self.temporal_kernel_size,
            spatial_kernel_size,
            A.size(1))
        self.data_bn = nn.BatchNorm2d(in_channels)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            DRGCBlock(in_channels,
                         64,
                         kernel_size,
                         1,
                         residual=False,
                         **kwargs0),
            DRGCBlock(64, 64, kernel_size, 1, **kwargs),
            DRGCBlock(64, 64, kernel_size, 1, **kwargs),
            DRGCBlock(64, 64, kernel_size, 1, **kwargs),
            DRGCBlock(64, 128, kernel_size, 2, **kwargs),
            DRGCBlock(128, 128, kernel_size, 1, **kwargs),
            DRGCBlock(128, 128, kernel_size, 1, **kwargs),
            DRGCBlock(128, 256, kernel_size, 2, **kwargs),
            DRGCBlock(256, 256, kernel_size, 1, **kwargs),
            DRGCBlock(256, 256, kernel_size, 1, **kwargs),
        ))

        self.RAMGen = RAMGen(
            3,
            RAM_encoder_output_channels,
            RAM_decoder_output_channels,
            kernel_size,
            T,
            self.relative_attention_component,
            self.geometric_component)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
            # edge importance for RAM_r's encoder and decoder in RAMGen
            self.RAMGen_edge_importance = nn.Parameter(
                torch.ones(self.A.size()))
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=(1, 1))

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        # batchnorm2d start
        x = x.contiguous().view(N, C, T, V*M)
        x = self.data_bn(x)
        x = x.view(N, C, T, V, M)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(N, M, C, T, V)
        # batchnorm2d end

        RAM = self.RAMGen(x, self.A*self.RAMGen_edge_importance)
        for gcn, importance in zip(
                self.st_gcn_networks, self.edge_importance):
            x, RAM, _ = gcn(x, RAM, self.A*importance)

        # global pooling
        N, M, C, T, V = x.shape
        x = x.contiguous().view(N*M, C, T, V)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x


class DRGCBlock(nn.Module):
    r"""Dyadic Relational Graph Convolution Block. Applies SGC and DRGC over
            input graph. Temporal convlution applies over SGC and DRGC's result
            and RAM.
        Input graph is divided into two subgraphs, physical graph
            and relational graphs. Physical graph is presented by a
            static adjacency matrix. Relational graphs are presented by RAM.
            Each relational graph is constructed for one corresponding frame.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the olution
        kernel_size (tuple):
            Size of the temporal kernel, SGC kernel, and V
        stride (int, optional): Stride of the temporal olution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional):
            If ``True``, applies a residual mechanism. Default: ``True``

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
            :math:`K` is the spatial kernel size,
                as :math:`K == kernel_size[1]`,
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

        assert len(kernel_size) == 3
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        RAM_padding = ((kernel_size[0]-1) // 2, 0)

        self.gcn = SpatialDyadicRelationalGC(
            in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        self.RAM_tcn = nn.Sequential(
          nn.Conv2d(
            1,
            1,
            (kernel_size[0], 1),
            (stride, 1),
            RAM_padding,
          ),
          nn.BatchNorm2d(1),
        )

        if not residual:
            self.residual = lambda x: 0
            self.RAM_res = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
            self.RAM_res = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
            self.RAM_res = nn.Sequential(
                nn.Conv2d(1,
                          1,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(1),
            )

        self.relu = nn.ReLU(inplace=True)

        # RAM_importance is denoted as \mu in our paper.
        self.RAM_importance = nn.Parameter(torch.ones((1, 1)))

    def forward(self, x, RAM, A):

        N, M, C, T, V = x.shape
        # res for tcn
        res = self.residual(x.contiguous().view(N*M, C, T, V))
        # SGC and DRGC
        x, _, A = self.gcn(x, RAM*self.RAM_importance, A)
        N, M, C, T, V = x.shape
        x = x.contiguous().view(N*M, C, T, V)
        x = self.tcn(x) + res
        _, C, T, V = x.shape
        x = x.contiguous().view(N, M, C, T, V)

        AN, AT, AV, AW = RAM.shape
        RAM = RAM.contiguous().view(AN, 1, AT, AV*AW)
        # tcn over RAM
        RAM = self.RAM_tcn(RAM)+self.RAM_res(RAM)
        AN, _, AT, _ = RAM.shape
        RAM = torch.squeeze(RAM, 1)
        RAM = RAM.contiguous().view(AN, AT, AV, AW)
        RAM = normalizeRAM(scaledTanh(RAM))

        return self.relu(x), RAM, A
