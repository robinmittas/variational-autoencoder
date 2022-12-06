import torch; torch.manual_seed(0)
from PIL import Image
import torch.utils
import torch.distributions
import torchvision
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from models.linear_variational_autoencoder import *
from models.linear_autoencoder import *
import os
import yaml



def train(autoencoder, training_data, validation_data, epochs=20, kld_loss_weight=0.00025):
    """
    A function to train the (variational) Autoencoder. If you want to train the VAE, set vaitational to True
    :param autoencoder: Neural Net, subclassed from nn.Moudle
    :param training_data: data in format x,y (input, class)
    :param epochs: numer of epochs
    :param variational: Boolean, if true we add KL divergence term to loss
    :return:
    """
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        running_loss = 0.0
        loss_epoch = []
        with tqdm(training_data, unit="batch") as tdata:
            for x, y in tdata:
                # set model to train mode
                autoencoder.train()
                x = x.to(device) # GPU
                opt.zero_grad()
                x_hat, x, mu, log_sigma = autoencoder(x)
                # TODO Hyperparameter kld loss weight!!!
                loss = autoencoder.loss(reconstruction=x_hat, input=x, mu=mu, log_sigma=log_sigma, kld_loss_weight=kld_loss_weight)
                # add loss per batch to list
                loss_epoch.append(loss["loss"].item())
                ## add Losses and accuracy measures to tensorpoard
                writer.add_scalar('Loss/train', loss["loss"], epoch)
                writer.add_scalar('Loss/KLD', loss["KLD"], epoch)
                writer.add_scalar('Loss/reconstruction_loss', loss["reconstruction_loss"], epoch)

                loss["loss"].backward()
                opt.step()

                # add to running loss
                running_loss += loss["loss"].item()

            final_loss = np.mean(loss_epoch)
            writer.add_scalar('Loss/epoch_loss', loss["reconstruction_loss"], epoch)

        # evaluate model - validation
        #eval_loss = evaluate_model(autoencoder, validation_data)
        #print(eval_loss)
        print("RUNNING LOSS:", running_loss)
    return autoencoder

def evaluate_model(model, validation_data):
    with torch.no_grad():
        model.eval()
        model.to(device)
        criterion = torch.nn.MSELoss()
        loss = 0
        for x,y in validation_data:
            x_hat = model(x)
            loss += criterion(
                x_hat,
                x
            ).item()
    return loss

"""
autoencoder = Autoencoder(latent_dims).to(device) # GPU

# use MNST Dataset
data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
               transform=torchvision.transforms.ToTensor(),
               download=True),
        batch_size=128,
        shuffle=True)

autoencoder = train(autoencoder, data)
"""
def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break

def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
"""
plot_reconstructed(autoencoder)



vae = VariationalAutoencoder(latent_dims).to(device) # GPU
vae = train(vae, data, variational=True)
plot_latent(vae, data)

plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))

def interpolate(autoencoder, x_1, x_2, n=12):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)
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
x, y = data.__iter__().next() # hack to grab a batch
x_1 = x[y == 1][1].to(device) # find a 1
x_2 = x[y == 0][1].to(device) # find a 0
interpolate(vae, x_1, x_2, n=20)

interpolate(autoencoder, x_1, x_2, n=20)


def interpolate_gif(autoencoder, filename, x_1, x_2, n=100):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)

    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])

    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()*255

    images_list = [Image.fromarray(img.reshape(28, 28)).resize((256, 256)) for img in interpolate_list]
    images_list = images_list + images_list[::-1] # loop back beginning

    images_list[0].save(
        f'{filename}.gif',
        save_all=True,
        append_images=images_list[1:],
        loop=1)
interpolate_gif(vae, "vae", x_1, x_2)
"""


def main(config_filename:str):
    # get configs from file
    # load config file
    with open(config_filename, encoding='utf8') as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)
    # use MNIST Dataset and load training and test data
    training_data = torchvision.datasets.MNIST(root='./data', transform=torchvision.transforms.ToTensor(), train=True, download=True)

    test_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data', transform=torchvision.transforms.ToTensor(), train=False, download=True),
        batch_size=128,
        shuffle=True
    )

    train_size = int(config["train_test_split"] * len(training_data))
    val_size = len(training_data) - train_size
    train_set, val_set = torch.utils.data.random_split(training_data, [train_size, val_size])

    # Load data into torch Dataloader
    train_set = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    val_set = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True)

    vae = VariationalAutoencoder(config["latent_dimension"])
    vae = train(vae, train_set, val_set, epochs=config["epochs"], kld_loss_weight=config["kld_loss_weight"])

    return vae


if __name__ == "__main__":
    # define some variables
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    latent_dims = 2
    writer = SummaryWriter("./logs_tensorboard/")
    vae = main("configs/config_linear_variational_autoencoder.yaml")

