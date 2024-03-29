from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms, utils, datasets
import pytorch_lightning as pl
from models.VarBayesianAE import *
from models.BetaVAE import *
from models.HEBAE import *
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
        print(self.params)
        self.save_hyperparameters()
        self.current_device = params["devices"]
        self.current_training_step = 0
        # copy is needed as otherwise kl weight is overwritten
        self.valid_params = params.copy()
        self.valid_params["KL_divergence_weight"] = 1
        self.valid_params["scale_kld"] = False
        print(self.params)


    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params["learning_rate"])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        self.current_training_step += 1
        x, y = train_batch
        self.current_device = x.device
        output = self.forward(x)
        loss = self.model.loss(output,
                               current_train_step=self.current_training_step,
                               **self.params)

        self.log_dict({f"train_{loss_key}": loss_val.item() for loss_key, loss_val in loss.items()})

        return loss["total_loss"]

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        self.current_device = x.device
        output = self.forward(x)
        loss = self.model.loss(output,
                               current_train_step=self.current_training_step,
                               **self.valid_params)  # for validation step set KLD weight always to 1

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
        standard_norm, sampled = self.model.sample(144, self.current_device)
        img_grid = utils.make_grid(sampled)
        tensorboard = self.logger.experiment
        tensorboard.add_image(f'Sampled Images {self.current_epoch}', img_grid)

    def on_train_end(self) -> None:
        standard_norm, sampled = self.model.sample(144, self.current_device)
        img_grid = utils.make_grid(sampled, nrow=12)
        vutils.save_image(img_grid.cpu().data,
                          self.params["plot_sample"],
                          normalize=True,
                          nrow=12)


if __name__ == "__main__":
    with open("configs/linear_vae_config.yaml", encoding='utf8') as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)
        conf.close()

    # transforms for training and validation sets
    transform = transforms.Compose([
        transforms.Resize((config["resize"], config["resize"])),
        transforms.ToTensor()
    ])

    # In case we want to train on MNIST, we use built in API, otherwise small workaround: Download CelebA Data and use ImageFolder instead
    if config["data_path"] == "MNIST":
        data = MNIST(root='./data', transform=transforms.ToTensor(), train=True, download=True)
        test_loader = torch.utils.data.DataLoader(
                                MNIST(root='./data', transform=transforms.ToTensor(), train=False, download=True),
                                batch_size=128,
                                shuffle=False)
    else:
        data = datasets.ImageFolder(config["data_path"], transform=transform)

    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size

    train, val = torch.utils.data.random_split(data, [train_size, val_size])

    if config["data_path"] != "MNIST":
        val_size_new = int(val_size/2)
        val, test = torch.utils.data.random_split(val, [val_size_new, val_size-val_size_new])
        test_loader = DataLoader(test, batch_size=config["batch_size"], shuffle=False)

    train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val, batch_size=config["batch_size"], shuffle=False)



    #vae = SigmaVAE(in_channels=config["input_image_size"][0],
    #               hidden_dimensions=config["hidden_dimensions"],
    #               latent_dimension=config["latent_dimension"],
    #               kernel_size=config["kernel_size"],
    #               stride=config["stride"],
    #               padding=config["padding"],
    #               max_pool=config["max_pool"],
    #               linear_layer_dimension=config["linear_layer_dimension"],
    #               last_conv_layer_kernel_size=config["last_conv_layer_kernel_size"])

    #vae = VarBayesianAE(in_channels=config["input_image_size"][0],
    #                    hidden_dimensions=config["hidden_dimensions"],
    #                    latent_dimension= config["latent_dimension"],
    #                    kernel_size=config["kernel_size"],
    #                    stride=config["stride"],
    #                    padding=config["padding"],
    #                    max_pool=config["max_pool"],
    #                    linear_layer_dimension=config["linear_layer_dimension"],
    #                    last_conv_layer_kernel_size=config["last_conv_layer_kernel_size"])

    #vae = BetaVAE(in_channels=config["input_image_size"][0],
    #              hidden_dimensions=config["hidden_dimensions"],
    #              latent_dimension=config["latent_dimension"],
    #              kernel_size=config["kernel_size"],
    #              stride=config["stride"],
    #              padding=config["padding"],
    #              max_pool=config["max_pool"],
    #              linear_layer_dimension=config["linear_layer_dimension"],
    #              last_conv_layer_kernel_size=config["last_conv_layer_kernel_size"])

    #vae = HEBAE(in_channels=config["input_image_size"][0],
    #            hidden_dimensions=config["hidden_dimensions"],
    #            latent_dimension=config["latent_dimension"],
    #            kernel_size=config["kernel_size"],
    #            stride=config["stride"],
    #            padding=config["padding"],
    #            max_pool=config["max_pool"],
    #            linear_layer_dimension=config["linear_layer_dimension"],
    #            last_conv_layer_kernel_size=config["last_conv_layer_kernel_size"])

    vae = LinearVAE(input_dimension=config["input_image_size"],
                    hidden_dimensions=config["hidden_dimensions"],
                    latent_dim=config["latent_dimension"])

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



