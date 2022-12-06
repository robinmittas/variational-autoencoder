import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # In case we have cuda enabled
        if device == "cuda":
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        log_sigma = self.linear3(x)#torch.log(self.linear3(x))
        z = mu + log_sigma * self.N.sample(mu.shape)
        # define KL Divergence as described in the Appendix XY TODO!
        self.kl = 0.5 * torch.sum(1+torch.log(log_sigma**2) - mu**2 - log_sigma**2)
        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        return mu, log_sigma


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def reparameterize(self, mu, log_sigma):
        """
        Reparameterization trick to sample from N(mu, var): See Appendix
        The Encoder returns mean and log(variance). We then sample from Standard Normal Distribution, multiply and add to retrieve
        exp(0.5*log_sigma)*N(0,1) + mu          ~ N(mu, sigma)
        which then behaves like ~ N(mu, sigma) and thus we have sampled from latent space.
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        z = self.reparameterize(mu, log_sigma)
        return self.decoder(z), x, mu, log_sigma

    def loss(self, reconstruction, input, mu, log_sigma, kld_loss_weight) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        #reconstruction =reconstruction
        #input = input
        #mu = args[2]
        #log_var = args[3]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        reconstruction_loss = F.mse_loss(reconstruction, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu ** 2 - log_sigma.exp(), dim = 1), dim = 0)

        loss = reconstruction_loss + kld_loss_weight * kld_loss
        return {'loss': loss, 'reconstruction_loss': reconstruction_loss.detach(), 'KLD': -kld_loss.detach()}
