import os 
import sys 
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser

import pytorch_lightning as pl

from dcgan.datasets.cifar10 import CIFAR10DataModule
from dcgan.datasets.mnist import MNISTDataModule 

from dcgan.models.dcgan import DCGAN

def main():
    args = get_args() 

    if args.dataset == "CIFAR10":
        dm = CIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == "MNIST":
        dm = MNISTDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise NotImplementedError("Dataset not supported! Should be one of: \n 1. CIFAR10 \n 2. MNIST")

    model = DCGAN(
        z_filter_shape=args.z_filter_shape,
        n_channels=args.n_channels,
        generator_lr=args.generator_lr,
        discriminator_lr=args.discriminator_lr,
        beta1=args.beta1
    )  

    trainer = pl.Trainer.from_argparse_args(args) 
    trainer.fit(model, dm) 

    trainer.test(dm)

def get_args():
    parser = ArgumentParser() 
    parser.add_argument("--dataset", default="CIFAR10", type=str) 
    parser.add_argument("--data_dir", default="./datasets/", type=str)
    parser.add_argument("--batch_size", default=128, type=int) 
    parser.add_argument("--num_workers", default=4, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DCGAN.add_model_specific_args(parser)
    args = parser.parse_args() 
    return args 

if __name__ == "__main__":
    main()

