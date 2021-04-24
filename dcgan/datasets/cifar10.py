import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data.dataset import random_split 
import pytorch_lightning as pl 


class CIFAR10Dataset(nn.Module): 

    def __init__(self, data_dir: str="./", transform=None, train=True):
        super().__init__()
        self.cifar10 = CIFAR10(data_dir, train=train, download=True)
        if transform is None: 
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform 

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        z = torch.randn(100)
        image, _ = self.cifar10[index]

        if self.transform: 
            image = self.transform(image)

        return image, z
        

class CIFAR10DataModule(pl.LightningDataModule): 

    def __init__(self, data_dir: str = "./", batch_size: int=128, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
            ])

    def setup(self, stage=None):
        self.test_set = CIFAR10Dataset(self.data_dir, transform=self.transform, train=False)
        full_dataset = CIFAR10Dataset(self.data_dir, transform=self.transform, train=True)
        self.train_set, self.val_set = random_split(full_dataset, [40000, 10000])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    dataset = CIFAR10Dataset()
    image, z = dataset[0]
    # check lengths
    print(image.shape, z.shape) 

    dm = CIFAR10DataModule()
    dm.setup() 

    # check lengths 
    print(len(dm.train_set), len(dm.test_set), len(dm.val_set))

    train_dataloader = dm.train_dataloader() 
    
    for image_batch, z_batch in train_dataloader:
        # check batch shapes
        print(image_batch.shape, z_batch.shape)
        break 