import torch
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision.utils as vutils


def interpolate_2_images(model,
                         test_data_loader,
                         label_1: int,
                         label_2: int,
                         device,
                         path,
                         n=12):
    """
    This function interpolates between two images: Both images are encoded, then the latent representations are interpolated with n steps
    :param model: VAE model (nn.Module subclassed)
    :param test_data_loader: Test Data Loader
    :param label_1: labels, e.g. 1 or 2
    :param label_2: labels
    :param device: device (gpu/ cpu)
    :param path: path to save image
    :param n: how many images we interpolate between those two
    :return:
    """
    x, y = next(iter(test_data_loader))  # hack to grab a batch
    x_1 = x[y == label_1][1].unsqueeze(dim=0)
    x_2 = x[y == label_2][1].unsqueeze(dim=0)
    z_1 = model.encoder(x_1.to(device))[0]
    z_2 = model.encoder(x_2.to(device))[0]
    z = torch.stack([z_1 + (z_2 - z_1) * t for t in np.linspace(0, 1, n)])
    interpolate_list = model.decoder(z)

    vutils.save_image(interpolate_list.cpu().data,
                      path,
                      normalize=True,
                      nrow=n)

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


def plot_2d_latent_space(autoencoder, r0=(-1, 1), r1=(-1, 1), n=50, input_dimension=[1, 28, 28]):
    """
    :param autoencoder:
    :param r0: borders for drawing from latent space
    :param r1: borders for drawing from latent space
    :param n: nxn grid
    :param input_dimension: input dimension of one image
    :return:
    """
    latent_space = torch.empty(n, n, input_dimension[0], input_dimension[1], input_dimension[2])
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]])#.to(device)
            latent_space[i, j] = autoencoder.decoder(z)
    latent_space = latent_space.cpu().data.view(-1, input_dimension[0], input_dimension[1], input_dimension[2])
    vutils.save_image(latent_space,
                      "./plots/2d_latent_space.png",
                      normalize=True,
                      nrow=n)
    return latent_space



"""
latent=plot_2d_latent_space(model.model, r0=(-1, 1), r1=(-1, 1),  n=50)

vutils.save_image(latent.cpu().data.view(-1,1,28,28),
                  "./plots/2d_latent_space.png.png",
                  normalize=True,
                  nrow=50)



from trainer_lightning_module import VAETrainer
import torch
import torchvision.utils as vutils

## check with other one
path = "C:\\Users\\robin\\Desktop\\MASTER Mathematics in Data Science\\Seminar\\variational-autoencoder\\logs\\SigmaVAE\\version_10\\checkpoints\\epoch=19-step=16879.ckpt"
model = VAETrainer.load_from_checkpoint(path)
checkpoint = torch.load(path)
checkpoint
model.load_state_dict(checkpoint["state_dict"])

samples = model.model.sample(144, "cpu")


vutils.save_image(samples.cpu().data,
                  "./plots/beta_vae_sampled.png",
                  normalize=True,
                  nrow=12)





label_1=1
label_2=3
n=12
x, y = next(iter(test_loader))  # hack to grab a batch
x_1 = x[y == label_1][1].unsqueeze(dim=0)
x_2 = x[y == label_2][1].unsqueeze(dim=0)
z_1 = model.model.encoder(x_1)[0]
z_2 = model.model.encoder(x_2)[0]
z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
interpolate_list = model.model.decoder(z)
#interpolate_list = interpolate_list.to('cpu').detach().numpy()

vutils.save_image(interpolate_list.cpu().data,
                  "./plots/standard_conv_vae_interpolate_2_numbers.png",
                  normalize=True,
                  nrow=n)
"""