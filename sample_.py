import torch

import yaml
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from trainer_lightning_module import *
import sys

sys.path.append("C:\\Users\\robin\\Desktop\\MASTER Mathematics in Data Science\\Seminar\\variational-autoencoder\\models")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("configs/var_bayesian_config.yaml", encoding='utf8') as conf:
    config = yaml.load(conf, Loader=yaml.FullLoader)
    conf.close()
# use MNIST Dataset and load training and test data
training_data = MNIST(root='./data', transform=transforms.ToTensor(), train=True, download=True)

test_data = torch.utils.data.DataLoader(
    MNIST(root='./data', transform=transforms.ToTensor(), train=False, download=True),
    batch_size=128,
    shuffle=True)

train_size = int(config["train_test_split"] * len(training_data))
val_size = len(training_data) - train_size
train_set, val_set = torch.utils.data.random_split(training_data, [train_size, val_size])

# Load data into torch Dataloader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)

# config contains further hyperparameters (LR/ KLD Weight/ MSE Reduction)
model = VAETrainer(model=vae, params=config)

vae = VarBayesianAE(in_channels=config["input_image_size"][0],
                    hidden_dimensions=config["hidden_dimensions"],
                    latent_dimension=config["latent_dimension"],
                    kernel_size=config["kernel_size"],
                    stride=config["stride"],
                    padding=config["padding"],
                    max_pool=config["max_pool"],
                    linear_layer_dimension=3)

path = "C:\\Users\\robin\\Desktop\\MASTER Mathematics in Data Science\\Seminar\\variational-autoencoder\\lightning_logs\\version_6\\checkpoints\\epoch=21-step=16500.ckpt"



## check with other one
path = "C:\\Users\\robin\\Desktop\\MASTER Mathematics in Data Science\\Seminar\\variational-autoencoder\\lightning_logs\\version_17\\checkpoints\\epoch=19-step=15000.ckpt"
model = VAETrainer.load_from_checkpoint(path)
checkpoint = torch.load(path)
checkpoint


## working
checkpoint = torch.load(path)
model.load_state_dict(checkpoint["state_dict"])

sample = torch.rand(16,16)
out = model.model.decoder(sample)

plt.imshow(out[2].permute(1,2,0).detach().numpy())

fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(16, 1))

for ax, im in zip(grid, out):
    # Iterating over the grid returns the Axes.
    ax.imshow(im.permute(1,2,0).detach().numpy())
    plt.xticks([])
    plt.yticks([])
plt.xticks([])
plt.yticks([])
plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_reconstructed(autoencoder, r0=(-10, 10), r1=(-10, 10), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])

plot_reconstructed(model.model, r0=(-3, 3), r1=(-3, 3),  n=10)

out = model.model.encoder(torch.unsqueeze(train_set[0][0], 0))

decoded = model.model.decoder(out[0])

plt.imshow(decoded[0].permute(1,2,0).detach().numpy())
plt.imshow(train_set[0][0].permute(1,2,0))



interpolate_a = 1
interpolate_b = 9
a = False
b = False
idx = 0
while not a or not b:
    if train_set[idx][1] == interpolate_a:
        a = True
        example_image_a = train_set[idx][0]
    elif train_set[idx][1] == interpolate_b:
        b = True
        example_image_b = train_set[idx][0]
    idx += 1

encode_a = model.model.encoder(example_image_a.unsqueeze(dim=0))
encode_b = model.model.encoder(example_image_b.unsqueeze(dim=0))


linfit = scipy.interpolate.interp1d([1, 9], np.vstack([encode_a[0].detach().numpy(), encode_b[0].detach().numpy()]), axis=0)
encoded_interpolate = linfit([i+1 for i in range(9)])

reconstructed = []
for sample in encoded_interpolate:
    rec = model.model.decoder(torch.from_numpy(sample).unsqueeze(dim=0).float())
    reconstructed.append(rec)

plt.imshow(reconstructed[7][0].permute(1,2,0).detach().numpy())


fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 3))

for ax, im in zip(grid, reconstructed):
    # Iterating over the grid returns the Axes.
    ax.imshow(im[0].permute(1,2,0).detach().numpy())

plt.show()

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)
    #

log_sigma = torch.zeros([])

batch_images = next(iter(val_loader))[0]
reconstructed_batch_images = model(batch_images)[0]

log_sigma = ((batch_images - reconstructed_batch_images) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()


rec = gaussian_nll(reconstructed_batch_images, log_sigma, batch_images).sum()

model.training_step(next(iter(val_loader)), batch_idx=1)

inputs =  model(batch_images)
model.model.loss(inputs)

loss(inputs)

def loss(inputs: list, **kwargs) -> dict:
    """
    The loss function for Variational Bayesian AE
    :param inputs: [reconstruction, orig_input, latent_sample, mu, log_sigma], which is exactly the output of forward pass
    :param kwargs: We need following two parameters: "KL_divergence_weight" and which MSE Loss to use: "mse_reduction"
    :return:
    """
    reconstruction, orig_input, latent_sample, mu, log_sigma = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]

    # get optimal Sigma as discribed here: https://github.com/orybkin/sigma-vae-pytorch/blob/4748a3ac1686316292607f40192c7ec2ded09893/model.py#L60
    log_sigma = ((reconstruction - orig_input) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()

    # reconstruction loss as sum over all pixels
    reconstruction_loss = gaussian_nll(reconstruction, orig_input,
                                            log_sigma).sum()  # nn.functional.mse_loss(reconstruction, orig_input, reduction=kwargs["mse_reduction"])
    # for derivation of KL Term of two Std. Normals, see Appendix TODO!
    KL_divergence_loss = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu ** 2 - log_sigma.exp(), dim=1), dim=0)


    # total loss is sum of reconstruction and KLD (NO Weight for KLD!)
    total_loss = reconstruction_loss + KL_divergence_loss

    return {"total_loss_TESTS": total_loss,
            "kl_divergence_loss": kl,
            "reconstruction_loss": reconstruction_loss}


x, y = next(iter(test_data)) #.__iter__().next() # hack to grab a batch
x_1 = x[y == 9][1]#.to(device) # find a 1
x_2 = x[y == 8][1]#.to(device) # find a 0

def interpolate(autoencoder, x_1, x_2, n=12):
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

interpolate(model.model, x_1.unsqueeze(dim=1), x_2.unsqueeze(dim=1), n=20)


def plot_from_distribution(model,
                           test_data_loader,
                           nrow_ncols: (int, int),
                           label: int,
                           noise: float = 0.1):
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

plot_from_distribution(model, test_data, (7,7), 3, 0.01)

x, y = next(iter(test_data))
x1 = x[y == 2][1]
x, y = next(iter(test_data))
x2 = x[y == 2][1]

x1_enc = model.model.encoder(x1.unsqueeze(dim=0))
x2_enc = model.model.encoder(x2.unsqueeze(dim=0))

standard_gauss = torch.randn_like(x_1_latent[0])
sample_latent = x_1_latent[1] + standard_gauss * x_1_latent[2].exp()

sample_construct = model.model.decoder(standard_gauss)[0]
plt.imshow(sample_construct.permute(1,2,0).detach().numpy())


x_1_latent = 0.5+x_1_latent
decoded = model.model.decoder(x_1_latent)[0]

plt.imshow(decoded.permute(1,2,0).detach().numpy())




x1_enc = model.model.encoder(x1.unsqueeze(dim=0))
x2_enc = model.model.encoder(x2.unsqueeze(dim=0))
fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(6, 6))

for i in range(8):
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(6, 6))
    for d, ax in zip(np.linspace(-2,2,36), grid):
        print(d)
        moved_tensor = torch.zeros(x1_enc[0].size())
        moved_tensor[0][i] = d
        x1_enc_moved = x1_enc[0] + moved_tensor
        x1_dec = model.model.decoder(x1_enc_moved)[0]
        ax.imshow(x1_dec.permute(1, 2, 0).detach().numpy())
    plt.show()



first_dim_latent = torch.stack(x1_enc)
