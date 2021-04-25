import os 
import sys 
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl

from dcgan.datasets.cifar10 import CIFAR10DataModule
from dcgan.datasets.mnist import MNISTDataModule 

from dcgan.models.dcgan import DCGAN

def main() -> None:
    args = get_args() 

    if args.dataset == "CIFAR10":
        dm = CIFAR10DataModule(data_dir=args.data_dir, image_size=args.image_size, batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == "MNIST":
        dm = MNISTDataModule(data_dir=args.data_dir, image_size=args.image_size, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise NotImplementedError("Dataset not supported! Should be one of: \n 1. CIFAR10 \n 2. MNIST")

    model = DCGAN(
        z_dim=args.z_dim,
        z_filter_shape=args.z_filter_shape,
        n_channels=args.n_channels,
        learning_rate=args.learning_rate,
        beta1=args.beta1
    )  

    trainer = pl.Trainer.from_argparse_args(args) 
    trainer.fit(model, dm) 

    trainer.test(datamodule=dm)

    show_batch(dm, model, args.z_dim)

def show_batch(dm: pl.LightningDataModule, model: pl.LightningModule, z_dim: int) -> None:
    """
    Plots and saves a batch of generated images from the generator of the trained DCGAN.

    Args:
        dm (pl.LightningDataModule): The data module used for training the DCGAN. 
        model (pl.LightningModule): The DCGAN model.
        z_dim (int): The size of noise vector used during DCGAN training.

    """
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 10, 10
    z = torch.randn(100, z_dim, device=model.device)
    images = model(z)
    images = denormalize(images)
    images = images.permute(0, 2, 3, 1)
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        image = images[i-1].detach().to("cpu").numpy()
        if image.shape[-1] == 1:
            plt.imshow(image, "gray")
        else:
            plt.imshow(image)

    if not os.path.exists(f"../results/{dm}/"):
        os.makedirs(f"./results/{dm}/")

    figure.savefig(f"../results/{dm}/result.png")
    plt.show()

def denormalize(image: torch.Tensor) -> torch.Tensor:
    """
    Denormalizes image back to 0-255 range and converts the image to the torch.uint8 dtype.

    Args: 
        image (torch.Tensor): Input image.

    Returns:
        image (torch.Tensor): The denormalized (0-255) image.
    """
    image = (image - image.min()) / (image.max() - image.min()) * 255
    return image.to(torch.uint8)

def get_args() -> Namespace:
    parser = ArgumentParser() 
    parser.add_argument("--dataset", default="CIFAR10", type=str) 
    parser.add_argument("--data_dir", default="./datasets/", type=str)
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--batch_size", default=128, type=int) 
    parser.add_argument("--num_workers", default=4, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DCGAN.add_model_specific_args(parser)
    args = parser.parse_args() 
    return args 

if __name__ == "__main__":
    main()