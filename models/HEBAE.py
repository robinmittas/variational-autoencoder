## Hierarchical Empirical Bayes Autoencoders: https://arxiv.org/pdf/2007.10389.pdf
import torch

from models.VarBayesianAE import *

class HEBAE(VarBayesianAE):
    def __init__(self,
                 in_channels: int,
                 hidden_dimensions,
                 latent_dimension: int,
                 kernel_size,
                 stride,
                 padding: int,
                 max_pool,
                 linear_layer_dimension: int,
                 last_conv_layer_kernel_size: [int, ...]):
        super(HEBAE, self).__init__(in_channels=in_channels,
                                    hidden_dimensions=hidden_dimensions,
                                    latent_dimension=latent_dimension,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    max_pool=max_pool,
                                    linear_layer_dimension=linear_layer_dimension,
                                    last_conv_layer_kernel_size=last_conv_layer_kernel_size)
        # overwrite reparametrization trick of encoder with slightly different one
        self.encoder.reparameterize = self.reparameterize

    def reparameterize(self, mu: torch.Tensor, logsigma: torch.Tensor):
        """
        Slightly different reparametrization
        :return:
        """
        # first get the covariance matrix of mus
        cov = torch.cov(mu.T)
        std = torch.exp(torch.mul(0.5, logsigma))
        # retrieve eps ~ N(0, 1)
        eps = torch.randn_like(std)
        # reparamterize
        return mu + eps * std * torch.sqrt(torch.diag(cov, 0))

    def loss(self, inputs: list, **kwargs):
        # loss is defined her in tensorflow https://github.com/ramachandran-lab/HEBAE/blob/master/CelebA/CelebA_HEBAE.py

        reconstruction, orig_input, latent_sample, mu, log_sigma = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        # first get the covariance matrix of mus
        cov = torch.cov(mu.T)
        # KL without constraint
        logdet = torch.logdet(cov)
        if torch.isnan(logdet):
            kl = 0.5 * (torch.trace(cov) - self.latent_dimension)
        else:
            kl = 0.5 * (torch.trace(cov) - self.latent_dimension - logdet)
        # constraint
        mean_mu = torch.mean(mu, dim=0)
        # loss_encoder is equal to kl of mu with N(0,I) = kl + constraint
        kl += 0.5 * torch.sum(mean_mu**2)

        reconstruction_loss = torch.mean(torch.pow(reconstruction-orig_input, 2))

        total_loss = kwargs["KL_divergence_weight"] * kl + reconstruction_loss

        return {"total_loss": total_loss,
                "kl_divergence_loss": kl,
                "reconstruction_loss": reconstruction_loss}


