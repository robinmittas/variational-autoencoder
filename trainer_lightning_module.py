import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms, utils
import pytorch_lightning as pl
from models.VarBayesianAE import *
from models.BaseVarAutoencoder import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import yaml

class VAETrainer(pl.LightningModule):
    def __init__(self, model: BaseVarAutoencoder, params):
        super().__init__()
        self.model = model
        self.params = params

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params["learning_rate"])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self.forward(x)
        loss = self.model.loss(output,
                               mse_reduction=self.params["mse_reduction"],
                               KL_divergence_weight=self.params["KL_divergence_weight"])
        self.log('train_total_loss', loss["total_loss"])
        self.log('train_reconstruction_loss', loss["reconstruction_loss"])
        self.log('train_kld_loss', loss["kl_divergence_loss"])
        return loss["total_loss"]

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        output = self.forward(x)
        loss = self.model.loss(output,
                               mse_reduction=self.params["mse_reduction"],
                               KL_divergence_weight=self.params["KL_divergence_weight"])
        self.log('valid_total_loss', loss["total_loss"])
        self.log('valid_reconstruction_loss', loss["reconstruction_loss"])
        self.log('valid_kld_loss', loss["kl_divergence_loss"])
        # Log reconstructed images!
        tensorboard = self.logger.experiment
        img_grid = utils.make_grid(output[0])
        tensorboard.add_image(f'Reconstructed Images {self.current_epoch}', img_grid)
        # It seems like validation batches are shuffled
        img_grid = utils.make_grid(output[1])
        tensorboard.add_image(f'Original Images {self.current_epoch}', img_grid)

        return loss["total_loss"]





if __name__ == "__main__":
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

    vae = VarBayesianAE(in_channels=config["input_image_size"][0],
                        hidden_dimensions=config["hidden_dimensions"],
                        latent_dimension= config["latent_dimension"],
                        kernel_size=config["kernel_size"],
                        stride= config["stride"],
                        padding=config["padding"],
                        max_pool=config["max_pool"],
                        linear_layer_dimension=3)
    parameter = {"learning_rate": config["learning_rate"],
                 "mse_reduction": config["mse_reduction"],
                 "KL_divergence_weight": config["KL_divergence_weight"]}

    model = VAETrainer(model=vae, params=parameter)
    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="valid_total_loss", mode="min")])
    trainer.fit(model, train_loader, val_loader)

    """
    # data
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])
    
    train_loader = DataLoader(mnist_train, batch_size=32)
    val_loader = DataLoader(mnist_val, batch_size=32)
    
    # model
    model = LitAutoEncoder()
    
    # training
    trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
    trainer.fit(model, train_loader, val_loader)
"""
