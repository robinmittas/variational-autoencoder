import torch
from torch.distributions import Bernoulli

from biva import DenseNormal, ConvNormal
from biva import VAE, LVAE, BIVA

# build a 2 layers VAE for binary images

# define the stochastic layers
z = [
    {'N': 8, 'kernel': 5, 'block': ConvNormal},  # z1
    {'N': 16, 'block': DenseNormal}  # z2
]

# define the intermediate layers
# each stage defines the configuration of the blocks for q_(z_{l} | z_{l-1}) and p_(z_{l-1} | z_{l})
# each stage is defined by a sequence of 3 resnet blocks
# each block is degined by a tuple [filters, kernel, stride]
stages = [
    [[64, 3, 1], [64, 3, 1], [64, 3, 2]],
    [[64, 3, 1], [64, 3, 1], [64, 3, 2]]
]

# build the model
model = VAE(tensor_shp=(-1, 1, 28, 28), stages=stages, latents=z)
biva = BIVA(tensor_shp=(-1, 1, 28, 28))
# forward pass and data-dependent initialization
x = torch.empty((8, 1, 28, 28)).uniform_().bernoulli()
data = model(x)  # data = {'x_' : p(x|z), z \sim q(z|x), 'kl': [kl_z1, kl_z2]}

# sample from prior
data = model.sample_from_prior(N=16)  # data = {'x_' : p(x|z), z \sim p(z)}
samples = Bernoulli(logits=data['x_']).sample()

# Calculate DB/EH
df["db_eh"] = np.where(df.sales_channel == "GK Agentur kundenbelegt",
                       df["bruttoumsatz_cars"] + df["bruttoumsatz_opt"] + df["hk_cars_adj"] + df["hk_opt_adj"] + df[
                           "andere_hk_cars_adj"] + df["off_standard_cars_adj"] + df["aufloesung_prap_cars_adj"] + df[
                           "plop_cars_adj"] + df["sekov_cars_adj"] + df["vkh_cars_adj"] + df["vkh_opt_adj"] + df[
                           "bonus_cars_adj"] + df["commisions_cars_adj"] + df["oth_cars_adj"] + df["oth_option_adj"] -
                       df["subvention_wasi"] - df["subvention_kif"] - df["rw_buchung"],
                       df["bruttoumsatz_cars"] + df["bruttoumsatz_opt"] + df["hk_cars_adj"] + df["hk_opt_adj"] + df[
                           "andere_hk_cars_adj"] + df["off_standard_cars_adj"] + df["aufloesung_prap_cars_adj"] + df[
                           "plop_cars_adj"] + df["sekov_cars_adj"] + df["vkh_cars_adj"] + df["vkh_opt_adj"] + df[
                           "oth_cars_adj"] + df["oth_option_adj"])

# Calculate CoR
df["cor"] = np.where(df.sales_channel == "GK Agentur kundenbelegt",
                     (df["wap_delta_adj"] - df["commisions_cars_adj"] + df["subvention_wasi"] + df["subvention_kif"] +
                      df["rw_buchung"] - df["sekov_cars_adj"] - df["vkh_cars_adj"] - df["vkh_opt_adj"] - df[
                          "bonus_cars_adj"]) / (df["upe_adj"]),
                     (df["wap_delta_adj"] - df["commisions_cars_adj"] - df["sekov_cars_adj"] - df["vkh_cars_adj"] - df[
                         "vkh_opt_adj"] - df["bonus_cars_adj"]) / (df["upe_adj"]))
