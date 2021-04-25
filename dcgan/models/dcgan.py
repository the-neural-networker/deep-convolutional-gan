from argparse import ArgumentParser

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import pytorch_lightning as pl  

from typing import Tuple, Dict


class Generator(nn.Module):
    """
    DCGAN Generator. Takes an noise vector of size z_dim and outputs 
    an image of shape (z_filter_shape * 16, z_filter_shape * 16, out_channels).

    Args:
        z_dim (int): Size of the input noise vector.
        z_filter_shape (int): Shape of the first convolutional filter. Output image will have height and width equal to z_filter_shape * 16. Set accordingly.
        out_channels (int): Number of channels in the output image. 

    Example: 
    ```
        generator = Generator(z_dim=100, z_filter_shape=2, out_channels=3)
        z = torch.randn(100).unsqueeze(0)
        image = generator(z)
    ```
    """

    def __init__(self, z_dim: int=100, z_filter_shape: int=2, out_channels: int=3) -> None:
        super(Generator, self).__init__()
        self.z_filter_shape = z_filter_shape
        self.fc = nn.Linear(z_dim, z_filter_shape * z_filter_shape * 1024) 
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.up4 = nn.ConvTranspose2d(128, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.apply(self._init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view((-1, 1024, self.z_filter_shape, self.z_filter_shape))
        x = F.relu(self.bn1(self.up1(x)))
        x = F.relu(self.bn2(self.up2(x))) 
        x = F.relu(self.bn3(self.up3(x))) 
        out = torch.tanh(self.up4(x))
        return out 

    def _init_weights(self, m: nn.Module) -> None:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if type(m) == nn.BatchNorm2d:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.bias)


class Discriminator(nn.Module): 
    """
    DCGAN Discriminator. Takes an image of shape (z_filter_shape * 16, z_filter_shape * 16, in_channels) 
    and outputs a probability of whether the image is real of fake.

    Args:
        z_filter_shape (int): The shape of the last convolutional filter before the sigmoid operation. Input image must then have height and width equal to z_filter_shape * 16.
        in_channels (int): The number of channels in the input image.

    Example: 
    ```
        image = ... # load image 
        discriminator = Discriminator(z_filter_shape=2, in_channels=3)
        out = discriminator(image) # image should be a tensor of shape (batch_size, in_channels, z_filter_shape * 16, z_filter_shape * 16)
    ```
    """

    def __init__(self, z_filter_shape: int=2, in_channels: int=3) -> None:
        super(Discriminator, self).__init__() 
        self.z_filter_shape = z_filter_shape
        self.down1 = nn.Conv2d(in_channels, 128, kernel_size=5, stride=2, padding=2)
        self.down2 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2) 
        self.bn1 = nn.BatchNorm2d(256) 
        self.down3 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2) 
        self.bn2 = nn.BatchNorm2d(512)
        self.down4 = nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2) 
        self.bn3 = nn.BatchNorm2d(1024) 
        self.fc = nn.Linear(z_filter_shape * z_filter_shape * 1024, 1)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.down1(x), negative_slope=0.2)
        x = F.leaky_relu(self.bn1(self.down2(x)), negative_slope=0.2) 
        x = F.leaky_relu(self.bn2(self.down3(x)), negative_slope=0.2) 
        x = F.leaky_relu(self.bn3(self.down4(x)), negative_slope=0.2) 
        x = x.view(-1, self.z_filter_shape * self.z_filter_shape * 1024)
        x = torch.sigmoid(self.fc(x))
        return x

    def _init_weights(self, m: nn.Module) -> None:
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if type(m) == nn.BatchNorm2d:
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
            nn.init.zeros_(m.bias)


class DCGAN(pl.LightningModule): 
    """
    Deep Convolutional Generative Adversarial Network. Trains a min-max game between the generator and the discriminator. 

    Args:
        z_dim (int): Size of the input noise vector.
        z_filter_shape (int): Shape of the first convolutional filter. Output image of the generator will have height and width equal to z_filter_shape * 16. Discriminator will take an input image of same dimensions.
        n_channels (int): Number of channels in the output image of the generator. (or) Number of channels of the input image to the discriminator.
        learning rate (float): Learning rate for the generator and discriminator optimizers. 
        beta1 (float): beta1 value for the Adam optimizer.

    Example: 
    ```
    dataloader = ... # create a dataloader of your training set. 
    model = DCGAN(z_dim=100, z_filter_shape=4, n_channels=3, learning_rate=2e-4, beta1=0.5)

    trainer = pl.Trainer() # lightning trainer. 
    trainer.fit(model, dataloader) # train the DCGAN

    z = torch.randn(100).unsqueeze(0) # noise vector of size z_dim
    generated_image = model(z)
    ```
    """

    def __init__(self, z_dim: int=100, z_filter_shape: int=2, n_channels: int=3, learning_rate: float=2e-4, beta1: float=0.5) -> None:
        super(DCGAN, self).__init__()
        self.save_hyperparameters()
        self.generator = Generator(self.hparams.z_dim, self.hparams.z_filter_shape, self.hparams.n_channels)
        self.discriminator = Discriminator(self.hparams.z_filter_shape, self.hparams.n_channels) 

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        image = self.generator(z) 
        return image 

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        x, _ = batch 
        batch_size = len(x)
        z = self._sample_z(batch_size, self.hparams.z_dim)

        if optimizer_idx == 0:
            x_hat = self(z) 
            y_hat = self.discriminator(x_hat) 

            loss = self.generator_loss(y_hat)
            self.log("train_gen_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if optimizer_idx == 1:
            x_hat = self(z) 
            y = self.discriminator(x) 
            y_hat = self.discriminator(x_hat) 

            loss = self.discriminator_loss(y, y_hat)
            self.log("train_disc_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, _ = batch 
        batch_size = len(x)
        z = self._sample_z(batch_size, self.hparams.z_dim)

        x_hat = self(z) 
        y = self.discriminator(x) 
        y_hat = self.discriminator(x_hat) 

        gen_loss = self.generator_loss(y_hat)
        self.log("val_gen_loss", gen_loss, on_step=True, on_epoch=True, prog_bar=True)
        disc_loss = self.discriminator_loss(y, y_hat)
        self.log("val_disc_loss", disc_loss, on_step=True, on_epoch=True, prog_bar=True)

        loss = {
            "gen_loss": gen_loss, 
            "disc_loss": disc_loss
        }

        return loss

    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, _ = batch 
        batch_size = len(x)
        z = self._sample_z(batch_size, self.hparams.z_dim)

        x_hat = self(z) 
        y = self.discriminator(x) 
        y_hat = self.discriminator(x_hat) 

        gen_loss = self.generator_loss(y_hat)
        self.log("test_gen_loss", gen_loss, on_step=True, on_epoch=True, prog_bar=True)
        disc_loss = self.discriminator_loss(y, y_hat)
        self.log("test_disc_loss", disc_loss, on_step=True, on_epoch=True, prog_bar=True)

        loss = {
            "gen_loss": gen_loss, 
            "disc_loss": disc_loss
        }

        return loss

    def generator_loss(self, y_hat: torch.Tensor) -> torch.Tensor: 
        return F.binary_cross_entropy(y_hat, torch.ones_like(y_hat))
    
    def discriminator_loss(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(y, torch.ones_like(y)) + F.binary_cross_entropy(y_hat, torch.zeros_like(y_hat))

    def configure_optimizers(self) -> Tuple[optim.Optimizer]:
        optimizer1 = optim.Adam(self.generator.parameters(), lr=self.hparams.learning_rate, betas=(self.hparams.beta1, 0.999))
        optimizer2 = optim.Adam(self.discriminator.parameters(), lr=self.hparams.learning_rate, betas=(self.hparams.beta1, 0.999))
        return optimizer1, optimizer2

    def _sample_z(self, batch_size: int, dim: int) -> torch.Tensor:
        return torch.randn(batch_size, dim, device=self.device)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--z_dim", type=int, default=100)
        parser.add_argument('--z_filter_shape', type=int, default=2)
        parser.add_argument('--n_channels', type=int, default=3)
        parser.add_argument('--learning_rate', type=float, default=2e-4)
        parser.add_argument('--beta1', type=float, default=0.5)
        return parser

if __name__ == "__main__":
    generator = Generator(z_filter_shape=2)
    z = torch.randn(100).unsqueeze(0)
    image = generator(z) 
    print(image.shape)

    discriminator = Discriminator(z_filter_shape=2)
    out = discriminator(image)
    print(out)

    dcgan = DCGAN()
    print(dcgan)