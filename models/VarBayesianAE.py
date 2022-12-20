import torch.nn.functional
from models.BaseVarAutoencoder import *

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
        """
        Variational Bayesian Encoder
        :param in_channels: Define in Channels of Input image (e.g. for [1,28,28] --> 1)
        :param hidden_dimensions: define hidden dimensions for Convolutional Blocks: list of ints [32,64,128..]
        :param latent_dimension: The dimension of the latent space of which we will sample
        :param kernel_size: Kernel size of Conv Layers (e.g. (2,2) or (3,3)
        :param stride: Stride for Conv Layers (tuple of ints)
        :param padding: Padding for both [H,W]
        :param max_pool: list of boolean (needs to have same length as @hidden_dimensions). Defines wether to use Max Pool in Conv Block
        :param linear_layer_dimension: The input dimension for last linear layer (e.g. the output dimension of (H or W) of last Conv Block)
        """

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
        """
        Forward Method for input batch-image
        :param x: batch
        :return: mu, log(sigma) and sample from N(mu, sigma)
        """
        out = self.encoder(x)
        # get out tensor into [B, x] shape for linear Layers
        out = torch.flatten(out, start_dim=1)
        mu = self.linear1(out)
        logsigma = self.linear2(out)
        return [self.reparameterize(mu, logsigma), mu, logsigma]

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
                 linear_layer_dimension: int,
                 last_conv_layer_kernel_size: [int, ...]):
        """
        Variational Bayesian Decoder
        :param in_channels: Define in Channels of initial Input image (e.g. for [1,28,28] --> 1)
        :param hidden_dimensions: define hidden dimensions for Transp-Convolutional Blocks: list of ints [32,64,128..]
        :param latent_dimension: The dimension of the latent space of which we will sample
        :param kernel_size: Kernel size of Transp-Conv Layers (e.g. (2,2) or (3,3)
        :param stride: Stride for Transp-Conv Layers (tuple of ints)
        :param padding: Padding for both [H,W]
        :param upsample: list of boolean (needs to have same length as @hidden_dimensions). Defines wether to use Upsample in Transp-Conv Block
        :param linear_layer_dimension: The input dimension for last linear layer (e.g. the output dimension of (H or W) of last Conv Block)
        """
        super(VarBayesianDecoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dimension = latent_dimension

        # we use same hidden dimension as encoder ([32,64,128...]) but as we now want to transpose/ decode, we reverse order
        hidden_dimensions.reverse()

        # first add linear layer to reshape sampled vector from latent space
        self.linear_dim = linear_layer_dimension
        self.hidden_dim_first = hidden_dimensions[0]
        self.linear = nn.Linear(latent_dimension, self.linear_dim**2 * self.hidden_dim_first)

        conv_blocks_decoder = []
        for idx, hidden_dim in enumerate(hidden_dimensions):
            if idx != len(hidden_dimensions)-1:
                in_channels = hidden_dim
                out_channels = hidden_dimensions[idx + 1]
            else:
                in_channels = hidden_dim
                out_channels = self.in_channels
            # Build Decoder
            conv_blocks_decoder.append(BaseTransposeConvBlock(in_channels=in_channels,
                                                              out_channels=out_channels,
                                                              kernel_size=kernel_size,
                                                              stride=stride,
                                                              padding=padding,
                                                              upsample=upsample[idx]))
        self.decoder = nn.Sequential(*conv_blocks_decoder)
        # add one last layer as heigt, weight of image is bigger then initally
        self.final_layer = nn.Conv2d(in_channels=self.in_channels,
                                     out_channels=self.in_channels,
                                     kernel_size=last_conv_layer_kernel_size,
                                     stride=(1, 1),
                                     padding=0)
        #self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> [torch.Tensor, ...]:
        """
        Forward method of Decoder
        :param x: Batch of samples from Latent Space
        :return: Batch of reconstructed Images
        """
        out = self.linear(x)
        # now reshape tensor of shape [B, hidden_dimensions[0] * linear_layer_dimension**2]
        out = out.view(-1, self.hidden_dim_first, self.linear_dim, self.linear_dim)
        out = self.decoder(out)
        out = self.final_layer(out)
        return self.sigmoid(out)


class VarBayesianAE(BaseVarAutoencoder):
    def __init__(self,
                 in_channels: int,
                 hidden_dimensions: [int, ...],
                 latent_dimension: int,
                 kernel_size: tuple[int, ...],
                 stride: tuple[int, ...],
                 padding: int,
                 max_pool: [bool, ...],
                 linear_layer_dimension: int,
                 last_conv_layer_kernel_size: [int, ...]):
        super(VarBayesianAE, self).__init__()

        self.latent_dimension = latent_dimension
        self.encoder = VarBayesianEncoder(in_channels=in_channels,
                                          hidden_dimensions=hidden_dimensions,
                                          latent_dimension=latent_dimension,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          max_pool=max_pool,
                                          linear_layer_dimension=linear_layer_dimension)

        self.decoder = VarBayesianDecoder(in_channels=in_channels,
                                          hidden_dimensions=hidden_dimensions,
                                          latent_dimension=latent_dimension,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          upsample=max_pool,
                                          linear_layer_dimension=linear_layer_dimension,
                                          last_conv_layer_kernel_size=last_conv_layer_kernel_size)

    def forward(self, inputs: torch.Tensor) -> [torch.Tensor, ...]:
        """
        Forward Method for Variational Autoencoder
        :param inputs: Takes a batch of images of inital size as an input [B,C,H,W]
        :return: [reconstruction, inputs, sample from latent space, mu, log_sigma]
        """
        encode = self.encoder(inputs)
        decode = self.decoder(encode[0])
        return [decode, inputs] + encode

    def loss(self, inputs: list, **kwargs) -> dict:
        """
        The loss function for Variational Bayesian AE
        :param inputs: [reconstruction, orig_input, latent_sample, mu, log_sigma], which is exactly the output of forward pass
        :param kwargs: We need following two parameters: "KL_divergence_weight" and which MSE Loss to use: "mse_reduction"
        :return:
        """
        reconstruction, orig_input, latent_sample, mu, log_sigma = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]

        reconstruction_loss = nn.functional.mse_loss(reconstruction, orig_input, reduction=kwargs["mse_reduction"])
        # for derivation of KL Term of two Std. Normals, see Appendix TODO!
        # KL_divergence_loss = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu ** 2 - log_sigma.exp(), dim=1), dim=0)
        KL_divergence_loss = -0.5 * torch.sum(1 + log_sigma - mu ** 2 - log_sigma.exp())

        # Add a weight to KL divergence term as the loss otherwise is too much dominated by this term!
        # For Validation we set this to 1
        KL_divergence_weight = kwargs["KL_divergence_weight"]

        total_loss = reconstruction_loss + KL_divergence_weight * KL_divergence_loss

        return {"total_loss": total_loss,
                "kl_divergence_loss": KL_divergence_loss,
                "reconstruction_loss": reconstruction_loss}






