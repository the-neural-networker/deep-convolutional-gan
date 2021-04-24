import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data.dataset import random_split 
import pytorch_lightning as pl
from torchvision.transforms.transforms import Resize 


class CIFAR10DataModule(pl.LightningDataModule): 

    def __init__(self, data_dir: str = "./", image_size: int=64, batch_size: int=128, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

    def __str__(self):
        return "cifar10"

    def setup(self, stage=None):
        self.test_set = CIFAR10(self.data_dir, download=True, transform=self.transform, train=False)
        full_dataset = CIFAR10(self.data_dir, download=True, transform=self.transform, train=True)
        self.train_set, self.val_set = random_split(full_dataset, [40000, 10000])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    dm = CIFAR10DataModule()
    dm.setup() 

    # check lengths 
    print(len(dm.train_set), len(dm.test_set), len(dm.val_set))

    train_dataloader = dm.train_dataloader() 
    
    for image_batch, z_batch in train_dataloader:
        # check batch shapes
        print(image_batch.shape, z_batch.shape)
        break 