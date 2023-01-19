from models.BaseVarAutoencoder import *
from models.VarBayesianAE import *

class BetaVAE(VarBayesianAE):
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
        super(BetaVAE, self).__init__(in_channels=in_channels,
                                      hidden_dimensions=hidden_dimensions,
                                      latent_dimension=latent_dimension,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      max_pool=max_pool,
                                      linear_layer_dimension=linear_layer_dimension,
                                      last_conv_layer_kernel_size=last_conv_layer_kernel_size)


    def loss(self, inputs: list, **kwargs) -> dict:
        """
        The loss function for Variational Bayesian AE
        :param inputs: [reconstruction, orig_input, latent_sample, mu, log_sigma], which is exactly the output of forward pass
        :param kwargs: We need following two parameters: "KL_divergence_weight" and which MSE Loss to use: "mse_reduction"
        :return:
        """
        reconstruction, orig_input, latent_sample, mu, log_sigma = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]

        reconstruction_loss = nn.functional.binary_cross_entropy(reconstruction, orig_input, reduction=kwargs["mse_reduction"])

        reconstruction_loss = reconstruction_loss / kwargs["batch_size"]

        # for derivation of KL Term of two Std. Normals, see Appendix TODO!
        KL_divergence_loss = -0.5 * torch.mean(1 + log_sigma - mu ** 2 - (log_sigma.exp())**2, dim=0)
        KL_divergence_loss = torch.sum(KL_divergence_loss)

        # Add a weight to KL divergence term as the loss otherwise is too much dominated by this term!
        # For Validation we set this to 1
        KL_divergence_weight = kwargs["KL_divergence_weight"]
        kld_factor = self.linear_scale_kld_factor(kwargs["current_train_step"],
                                                  kwargs["first_k_train_steps"],
                                                  0,
                                                  1) if kwargs["scale_kld"] else 1

        total_loss = reconstruction_loss + kld_factor * (KL_divergence_weight * KL_divergence_loss)


        return {"total_loss": total_loss,
                "kl_divergence_loss": KL_divergence_loss,
                "reconstruction_loss": reconstruction_loss}

    def linear_scale_kld_factor(self, current_training_step: int, first_k_train_steps: int, min_value=0, max_value=1):
        if first_k_train_steps == 0:
            return max_value
        delta = max_value - min_value
        scale_factor = min(min_value + delta * current_training_step/ first_k_train_steps, max_value)

        return scale_factor

