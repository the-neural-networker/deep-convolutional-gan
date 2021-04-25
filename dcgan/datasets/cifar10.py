import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data.dataset import random_split 

import pytorch_lightning as pl

from typing import Optional


class CIFAR10DataModule(pl.LightningDataModule): 
    """
    The CIFAR10 Data Module. Creates train, val and test dataloaders from the CIFAR10 Dataset.

    Args: 
        data_dir (str): Directory to which the CIFAR10 dataset will be downloaded.
        image_size (int): Size to which the images will be resized. 
        batch_size (int): Batch size of the train, val and test datalaoders. 
        num_workers (int): Number of workers to be used in the dataloaders.

    Example: 
    ```
        dm = CIFAR10DataModule(data_dir="/path/to/download/", image_size=32, batch_size=128, num_workers=4)
        dm.prepare_data()
        dm.setup()

        # iterate through batches.
        for image_batch, _ in dm.train_dataloader():
            ... # use batches for training.
    """
    def __init__(self, data_dir: str = "./", image_size: int=32, batch_size: int=128, num_workers=4) -> None:
        super(CIFAR10DataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

    def __str__(self) -> str:
        return "cifar10"

    def prepare_data(self) -> None:
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str]=None) -> None:
        
        if stage == 'fit' or stage is None:
            full_dataset = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.train_set, self.val_set = random_split(full_dataset, [40000, 10000])

        if stage == 'test' or stage is None:
            self.test_set = CIFAR10(self.data_dir, train=False, transform=self.transform)
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    dm = CIFAR10DataModule()
    dm.setup() 

    # check lengths 
    print(len(dm.train_set), len(dm.test_set), len(dm.val_set))

    train_dataloader = dm.train_dataloader() 
    
    for image_batch, _ in train_dataloader:
        # check batch shapes
        print(image_batch.shape)
        break 