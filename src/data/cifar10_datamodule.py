import os
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import lightning.pytorch as pl
from src.registry import register_datamodule

@register_datamodule("cifar10")
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, root:str, download:bool, mean, std, batch_size: int, num_workers: int, pin_memory: bool, prefetch_factor: int, persistent_workers: bool):
        super().__init__()
        self.root = root
        self.download = download
        self.mean, self.std = mean, std
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

    def prepare_data(self):
        datasets.CIFAR10(self.root, train=True,  download=self.download)
        datasets.CIFAR10(self.root, train=False, download=self.download)

    def setup(self, stage: Optional[str] = None):
        normalize = transforms.Normalize(self.mean, self.std)
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.ds_train = datasets.CIFAR10(self.root, train=True, transform=train_tf)
        self.ds_val = datasets.CIFAR10(self.root, train=False, transform=test_tf)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self): return self._loader(self.ds_train, shuffle=True)
    def val_dataloader(self):   return self._loader(self.ds_val,   shuffle=False)
    def test_dataloader(self):  return self._loader(self.ds_val,   shuffle=False)