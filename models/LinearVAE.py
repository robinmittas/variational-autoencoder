import numpy as np
import torch; torch.manual_seed(0)
import torch.nn.functional as F
import torch.utils
import torch.distributions
from models.BaseVarAutoencoder import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LinearVarEncoder(BaseEncoder):
    def __init__(self,
                 input_dimension: [int, ...],
                 hidden_dimensions:  [int, ...],
                 latent_dim):
        """
        :param input_dimension: [1,28,28] for mnist
        :param hidden_dimensions: [512,256,128...]
        :param latent_dim: int, e.g. 2
        """
        super(LinearVarEncoder, self).__init__()

        in_shape = np.prod(input_dimension)

        layers = [nn.Flatten(start_dim=1)]
        for hidden_dim in hidden_dimensions:
            layers.append(nn.Linear(in_shape, hidden_dim))
            layers.append(nn.ReLU())
            in_shape = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.linear1 = nn.Linear(hidden_dimensions[-1], latent_dim)
        self.linear2 = nn.Linear(hidden_dimensions[-1], latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.linear1(x)
        log_sigma = self.linear2(x)
        return [self.reparameterize(mu, log_sigma), mu, log_sigma]


class LinearDecoder(BaseDecoder):
    def __init__(self,
                 input_dimension: [int, ...],
                 hidden_dimensions:  [int, ...],
                 latent_dim):
        super(LinearDecoder, self).__init__()

        self.input_dimension = input_dimension
        # reverse hidden dim list
        hidden_dimensions.reverse()

        layers = []
        in_shape = latent_dim
        for hidden_dim in hidden_dimensions:
            layers.append(nn.Linear(in_shape, hidden_dim))
            layers.append(nn.ReLU())
            in_shape = hidden_dim

        # add last layer back to inital dimension
        layers.append(nn.Linear(hidden_dimensions[-1], np.prod(input_dimension)))

        self.decoder = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.decoder(z)
        out = self.sigmoid(out)
        return out.reshape((-1, self.input_dimension[0], self.input_dimension[1], self.input_dimension[2]))


class LinearVAE(BaseVarAutoencoder):
    def __init__(self,
                 input_dimension: [int, ...],
                 hidden_dimensions: [int, ...],
                 latent_dim
                 ):
        super(LinearVAE, self).__init__()
        self.encoder = LinearVarEncoder(input_dimension=input_dimension, hidden_dimensions=hidden_dimensions, latent_dim=latent_dim)
        self.decoder = LinearDecoder(input_dimension=input_dimension, hidden_dimensions=hidden_dimensions, latent_dim=latent_dim)
        self.latent_dimension = latent_dim

    def forward(self, inputs: torch.Tensor):
        encode = self.encoder(inputs)
        decode = self.decoder(encode[0])
        return [decode, inputs] + encode

    def loss(self, inputs: list, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param inputs: [reconstruction, orig_input, latent_sample, mu, log_sigma]
        :param kwargs:
        :return:
        """
        reconstruction, orig_input, latent_sample, mu, log_sigma = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        # use BCE or MSE
        reconstruction_loss = F.binary_cross_entropy(reconstruction, orig_input, reduction=kwargs["mse_reduction"])

        kld_loss = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())

        loss = reconstruction_loss + kwargs["KL_divergence_weight"] * kld_loss
        return {'total_loss': loss,
                'reconstruction_loss': reconstruction_loss.detach(),
                'kl_divergence_loss': kld_loss.detach()}
