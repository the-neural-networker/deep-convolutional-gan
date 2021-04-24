from argparse import ArgumentParser

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import pytorch_lightning as pl  


class Generator(nn.Module):

    def __init__(self, z_filter_shape=4, out_channels=3):
        super(Generator, self).__init__()
        self.z_filter_shape = z_filter_shape
        self.fc = nn.Linear(100, z_filter_shape * z_filter_shape * 1024) 
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.up4 = nn.ConvTranspose2d(128, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.apply(self._init_weights)

    def forward(self, z):
        x = self.fc(z)
        x = x.view((-1, 1024, self.z_filter_shape, self.z_filter_shape))
        x = F.relu(self.bn1(self.up1(x)))
        x = F.relu(self.bn2(self.up2(x))) 
        x = F.relu(self.bn3(self.up3(x))) 
        out = torch.tanh(self.up4(x))
        return out 

    def _init_weights(self, m):
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if type(m) == nn.BatchNorm2d:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)


class Discriminator(nn.Module): 

    def __init__(self, z_filter_shape=4, in_channels=3):
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

    def forward(self, x): 
        x = F.leaky_relu(self.down1(x), negative_slope=0.2)
        x = F.leaky_relu(self.bn1(self.down2(x)), negative_slope=0.2) 
        x = F.leaky_relu(self.bn2(self.down3(x)), negative_slope=0.2) 
        x = F.leaky_relu(self.bn3(self.down4(x)), negative_slope=0.2) 
        x = x.view(-1, self.z_filter_shape * self.z_filter_shape * 1024)
        x = torch.sigmoid(self.fc(x))
        return x

    def _init_weights(self, m):
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if type(m) == nn.BatchNorm2d:
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
            nn.init.zeros_(m.bias)


class DCGAN(pl.LightningModule): 
    """
    Deep Convolutional Generative Adversarial Network.
    """

    def __init__(self, z_filter_shape=4, n_channels=3, generator_lr=2e-4, discriminator_lr=2e-4, beta1=0.5):
        super(DCGAN, self).__init__()
        self.save_hyperparameters()
        self.generator = Generator(self.hparams.z_filter_shape, self.hparams.n_channels)
        self.discriminator = Discriminator(self.hparams.z_filter_shape, self.hparams.n_channels) 

    def forward(self, z):
        image = self.generator(z) 
        return image 

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, z = batch 
        if optimizer_idx == 0:
            x_hat = self.generator(z) 
            y_hat = self.discriminator(x_hat) 
            loss = self.generator_loss(y_hat)
            self.log("gen_loss", loss, on_epoch=False, prog_bar=True)
        if optimizer_idx == 1:
            x_hat = self.generator(z) 
            y = self.discriminator(x) 
            y_hat = self.discriminator(x_hat) 
            loss = self.discriminator_loss(y, y_hat)
            self.log("disc_loss", loss, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, z = batch 
        x_hat = self.generator(z) 
        y = self.discriminator(x) 
        y_hat = self.discriminator(x_hat) 
        gen_loss = self.generator_loss(y_hat)
        self.log("val_gen_loss", gen_loss, on_epoch=True, prog_bar=True)
        disc_loss = self.discriminator_loss(y, y_hat)
        self.log("val_disc_loss", disc_loss, on_epoch=True, prog_bar=True)
        return gen_loss, disc_loss

    def testing_step(self, batch, batch_idx):
        x, z = batch 
        x_hat = self.generator(z) 
        y = self.discriminator(x) 
        y_hat = self.discriminator(x_hat) 
        gen_loss = self.generator_loss(y_hat)
        self.log("loss/generator", gen_loss, on_epoch=True, prog_bar=True)
        disc_loss = self.discriminator_loss(y, y_hat)
        self.log("test_disc_loss", disc_loss, on_epoch=True, prog_bar=True)
        return gen_loss, disc_loss

    def generator_loss(self, y_hat, reduction="mean"): 
        loss = -torch.log(y_hat) 
        if reduction == "mean":
            return loss.mean() 
        else:
            return loss.sum() 
    
    def discriminator_loss(self, y, y_hat, reduction="mean"):
        loss = (torch.log(y) + torch.log(y_hat))
        if reduction == "mean":
            return -loss.mean() 
        else:
            return -loss.sum() 

    def configure_optimizers(self):
        optimizer1 = optim.Adam(self.generator.parameters(), lr=self.hparams.generator_lr, betas=(self.hparams.beta1, 0.999))
        optimizer2 = optim.Adam(self.discriminator.parameters(), lr=self.hparams.discriminator_lr, betas=(self.hparams.beta1, 0.999))
        return optimizer1, optimizer2

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--z_filter_shape', type=int, default=4)
        parser.add_argument('--n_channels', type=int, default=3)
        parser.add_argument('--generator_lr', type=float, default=2e-4)
        parser.add_argument('--discriminator_lr', type=float, default=2e-4)
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