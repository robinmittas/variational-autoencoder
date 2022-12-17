from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms, utils
import pytorch_lightning as pl
from models.VarBayesianAE import *
from models.LinearVAE import *
from models.SigmaVAE import *
from models.BaseVarAutoencoder import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
from plots import *


class VAETrainer(pl.LightningModule):
    def __init__(self, model: BaseVarAutoencoder, params):
        super(VAETrainer, self).__init__()
        self.model = model
        self.params = params
        self.save_hyperparameters()
        self.current_device = params["devices"]
        self.training_steps_total = 0

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params["learning_rate"])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        self.training_steps_total += 1
        x, y = train_batch
        self.current_device = x.device
        output = self.forward(x)
        loss = self.model.loss(output,
                               mse_reduction=self.params["mse_reduction"],
                               KL_divergence_weight=self.params["KL_divergence_weight"])

        self.log_dict({f"train_{loss_key}": loss_val.item() for loss_key, loss_val in loss.items()})

        return loss["total_loss"]

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        self.current_device = x.device
        output = self.forward(x)
        loss = self.model.loss(output,
                               mse_reduction=self.params["mse_reduction"],
                               KL_divergence_weight=1) # for validation step set KLD weight always to 1

        self.log_dict({f"valid_{loss_key}": loss_val.item() for loss_key, loss_val in loss.items()})

        # Log reconstructed validation images!
        tensorboard = self.logger.experiment
        img_grid = utils.make_grid(output[0])
        tensorboard.add_image(f'Reconstructed Images {self.current_epoch}', img_grid)
        # It seems like validation batches are shuffled
        img_grid = utils.make_grid(output[1])
        tensorboard.add_image(f'Original Images {self.current_epoch}', img_grid)

        return loss["total_loss"]

    def on_validation_epoch_end(self) -> None:
        """
        At the end of epoch we want to Sample from Latent Space and log these
        :return:
        """
        sampled = self.model.sample(144, self.current_device)
        img_grid = utils.make_grid(sampled)
        tensorboard = self.logger.experiment
        tensorboard.add_image(f'Sampled Images {self.current_epoch}', img_grid)

    def on_train_end(self) -> None:
        interpolate_2_images(self.model,
                             test_loader,
                             label_1=1,
                             label_2=3,
                             n=12,
                             device=self.current_device,
                             path=self.params["plot_2_interpolate_dir"])
        sampled = self.model.sample(144, self.current_device)
        img_grid = utils.make_grid(sampled, nrow=12)
        vutils.save_image(img_grid.cpu().data,
                          self.params["plot_sample"],
                          normalize=True,
                          nrow=12)


if __name__ == "__main__":
    with open("configs/beta_vae.yaml", encoding='utf8') as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)
        conf.close()
    # use MNIST Dataset and load training and test data
    training_data = MNIST(root='./data', transform=transforms.ToTensor(), train=True, download=True)

    test_loader = torch.utils.data.DataLoader(
        MNIST(root='./data', transform=transforms.ToTensor(), train=False, download=True),
        batch_size=128,
        shuffle=True)

    train_size = int(config["train_valid_split"] * len(training_data))
    val_size = len(training_data) - train_size
    train_set, val_set = torch.utils.data.random_split(training_data, [train_size, val_size])

    # Load data into torch Dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)

    #vae = SigmaVAE(in_channels=config["input_image_size"][0],
    #               hidden_dimensions=config["hidden_dimensions"],
    #               latent_dimension=config["latent_dimension"],
    #               kernel_size=config["kernel_size"],
    #               stride=config["stride"],
    #               padding=config["padding"],
    #               max_pool=config["max_pool"],
    #               linear_layer_dimension=config["linear_layer_dimension"])

    vae = VarBayesianAE(in_channels=config["input_image_size"][0],
                        hidden_dimensions=config["hidden_dimensions"],
                        latent_dimension= config["latent_dimension"],
                        kernel_size=config["kernel_size"],
                        stride=config["stride"],
                        padding=config["padding"],
                        max_pool=config["max_pool"],
                        linear_layer_dimension=config["linear_layer_dimension"])

    #vae = LinearVAE(input_dimension=config["input_image_size"],
    #                hidden_dimensions=config["hidden_dimensions"],
    #                latent_dim=config["latent_dim"])

    tb_logger = TensorBoardLogger(save_dir=config['logging_dir'],
                                  name=config['logging_name'])

    # config contains further hyperparameters (LR/ KLD Weight/ MSE Reduction)
    model = VAETrainer(model=vae, params=config)
    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="valid_total_loss", mode="min")],
                         logger=tb_logger,
                         max_epochs=config["epochs"],
                         accelerator=config["accelerator"],
                         devices=config["devices"])
    trainer.fit(model, train_loader, val_loader)



