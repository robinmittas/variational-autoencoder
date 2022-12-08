import torch
from BaseVarAutoencoder import *
from torch import nn


class VarBayesianEncoder(BaseEncoder):
    def __init__(self,
                 in_channels: int,
                 hidden_dimensions: [int, ...],
                 latent_dimension: int,
                 kernel_size: tuple[int, ...],
                 stride: tuple[int, ...],
                 padding: int,
                 max_pool: [bool, ...],
                 linear_layer_dimension: int):

        super(VarBayesianEncoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dimension = latent_dimension

        conv_blocks_encoder = []
        for idx, hidden_dim in enumerate(hidden_dimensions):
            # Build Encoder
            conv_blocks_encoder.append(BaseConvBlock(in_channels=in_channels,
                                                     out_channels=hidden_dim,
                                                     kernel_size=kernel_size,
                                                     stride=stride,
                                                     padding=padding,
                                                     max_pool=max_pool[idx]))
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*conv_blocks_encoder)
        # now add two dense layer to get mu and sigma from the latent space
        self.linear1 = nn.Linear(hidden_dimensions[-1] * linear_layer_dimension**2, latent_dimension)
        self.linear2 = nn.Linear(hidden_dimensions[-1] * linear_layer_dimension**2, latent_dimension)

    def forward(self, x: torch.Tensor) -> [torch.Tensor, ...]:
        out = self.encoder(x)
        # get out tensor into [B, x] shape for linear Layers
        out = torch.flatten(out, start_dim=1)
        mu = self.linear1(out)
        sigma = self.linear2(out)
        return [mu, sigma]

    def check_forward_shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        With this function you can check the dimensions/ shape of yout input tensor after the last convolutional block
        in order to define the input dimensions of the linear layers. For that first create an instance of class with
        dummy values.
        :param x: torch tensor of shape [B, C, H, W]
        :return: Tensor after conv layers (get shape by .size() or .shape
        """
        return self.encoder(x)


class VarBayesianDecoder(BaseDecoder):
    def __init__(self,
                 in_channels: int,
                 hidden_dimensions: [int, ...],
                 latent_dimension: int,
                 kernel_size: tuple[int, ...],
                 stride: tuple[int, ...],
                 padding: int,
                 upsample: [bool, ...],
                 linear_layer_dimension: int):
        super(VarBayesianDecoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dimension = latent_dimension

        conv_blocks_decoder = []
        for idx, hidden_dim in enumerate(hidden_dimensions):
            # Build Encoder
            conv_blocks_decoder.append(BaseTransposeConvBlock(in_channels=in_channels,
                                                              out_channels=hidden_dim,
                                                              kernel_size=kernel_size,
                                                              stride=stride,
                                                              padding=padding,
                                                              upsample=upsample[idx]))
            in_channels = hidden_dim

        self.decoder = nn.Sequential(*conv_blocks_decoder)
        # now add two dense layer to get mu and sigma from the latent space
        self.linear = nn.Linear(latent_dimension, hidden_dimensions[-1] * linear_layer_dimension)
        #self.linear2 = nn.Linear(hidden_dimensions[-1] * linear_layer_dimension, latent_dimension)

    def forward(self, x: torch.Tensor) -> [torch.Tensor, ...]:
        out = self.linear(x)
        out = self.decoder(x)
        return out

    def check_forward_shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        With this function you can check the dimensions/ shape of yout input tensor after the last convolutional block
        in order to define the input dimensions of the linear layers. For that first create an instance of class with
        dummy values.
        :param x: torch tensor of shape [B, C, H, W]
        :return: Shape after Conv Layers
        """
        return self.decoder(x).shape



class VarBayesianAE(BaseVarAutoencoder):
    def __init__(self):
        self.encoder = VarBayesianEncoder()
        self.decoder = VarBayesianDecoder()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.encoder(inputs)



encoder = VarBayesianEncoder(in_channels=1,
                  hidden_dimensions=[32,64,128],
                  latent_dimension= 128,
                  kernel_size=(2,2),
                  stride= (2,2),
                  padding=1,
                  max_pool=[False,False,False],
                             linear_layer_dimension=5)

mu, sigma = encoder(torch.rand([10,1,28,28]))#.shape

decoder = VarBayesianDecoder(in_channels=1,
                  hidden_dimensions=[32,64,128],
                  latent_dimension= 128,
                  kernel_size=(2,2),
                  stride= (2,2),
                  padding=1,
                  upsample=[False,False,False],
                  linear_layer_dimension=5)

decoder(encoder_output)