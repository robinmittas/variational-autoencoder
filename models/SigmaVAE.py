import torch.nn.functional
from models.VarBayesianAE import *
import numpy as np
import torch.nn.functional as F

class SigmaVAE(VarBayesianAE):
    def __init__(self,
                 in_channels: int,
                 hidden_dimensions: [int, ...],
                 latent_dimension: int,
                 kernel_size: tuple[int, ...],
                 stride: tuple[int, ...],
                 padding: int,
                 max_pool: [bool, ...],
                 linear_layer_dimension: int):

        super(SigmaVAE, self).__init__(in_channels=in_channels,
                                       hidden_dimensions=hidden_dimensions,
                                       latent_dimension=latent_dimension,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       max_pool=max_pool,
                                       linear_layer_dimension=linear_layer_dimension)

    def softclip(self, tensor, min):
        """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
        return min + F.softplus(tensor - min)

    def gaussian_nll(self, reconstruction: torch.Tensor, orig_input: torch.Tensor, sigma: torch.Tensor):
        return 0.5 * torch.pow((orig_input - reconstruction) / sigma.exp(), 2) + sigma + 0.5 * np.log(2 * np.pi)

    def loss(self, inputs: list, **kwargs) -> dict:
        """
        The loss function for Variational Bayesian AE
        :param inputs: [reconstruction, orig_input, latent_sample, mu, log_sigma], which is exactly the output of forward pass
        :param kwargs: We need following two parameters: "KL_divergence_weight" and which MSE Loss to use: "mse_reduction"
        :return:
        """
        reconstruction, orig_input, latent_sample, mu, log_sigma = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]

        # get optimal Sigma as discribed here: https://github.com/orybkin/sigma-vae-pytorch/blob/4748a3ac1686316292607f40192c7ec2ded09893/model.py#L60
        sigma = ((reconstruction - orig_input) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()
        # softclip the sigma
        sigma = self.softclip(sigma, -6)


        # reconstruction loss as sum over all pixels
        reconstruction_loss = self.gaussian_nll(reconstruction, orig_input, sigma).sum() #nn.functional.mse_loss(reconstruction, orig_input, reduction=kwargs["mse_reduction"])
        # for derivation of KL Term of two Std. Normals, see Appendix TODO!
        KL_divergence_loss = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())

        # total loss is sum of reconstruction and KLD (NO Weight for KLD!)
        total_loss = reconstruction_loss + KL_divergence_loss

        return {"total_loss": total_loss,
                "kl_divergence_loss": KL_divergence_loss,
                "reconstruction_loss": reconstruction_loss,
                "sigma": sigma}

