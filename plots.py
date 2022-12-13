import torch
import yaml
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from trainer_lightning_module import *
import torch


def interpolate_2_images(autoencoder,
                         test_data_loader,
                         label_1: int,
                         label_2: int,
                         n=12):
    x, y = next(iter(test_data_loader))  # hack to grab a batch
    x_1 = x[y == label_1][1]
    x_2 = x[y == label_2][1]
    z_1 = autoencoder.encoder(x_1)[0]
    z_2 = autoencoder.encoder(x_2)[0]
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    w = 28
    img = np.zeros((w, n*w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])


def plot_from_distribution(model,
                           test_data_loader,
                           nrow_ncols: (int, int),
                           label: int,
                           noise: float = 0.0):
    x, y = next(iter(test_data_loader))
    x = x[y == label][1]

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=nrow_ncols)

    x_1_latent = model.model.encoder(x.unsqueeze(dim=1))
    for ax in grid:
        standard_gauss = torch.randn_like(x_1_latent[1])
        sample_latent = x_1_latent[1] + standard_gauss * torch.exp(0.5 * x_1_latent[2])

        sample_latent += noise * torch.randn_like(x_1_latent[1])

        sample_construct = model.model.decoder(sample_latent)[0]

        #orig_z = model.model.decoder(x_1_latent[0])[0]
        # ax.imshow(orig_z.permute(1, 2, 0).detach().numpy())
        ax.imshow(sample_construct.permute(1, 2, 0).detach().numpy())
    plt.show()







