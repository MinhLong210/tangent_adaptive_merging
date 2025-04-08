import os
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        train_transform=None,
        test_transform=None,
    ):
        """
        CIFAR-10 DataModule for PyTorch Lightning.

        Args:
            root (str): Root directory where CIFAR-10 data will be stored/downloaded.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory. Defaults to False.
            train_transform (callable, optional): Transformations applied to training data.
            test_transform (callable, optional): Transformations applied to test data.
        """
        super().__init__()
        self.root = root
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        # Default transforms if none provided
        if self.train_transform is None:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        if self.test_transform is None:
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        # Initialize datasets
        self.train_dataset = datasets.CIFAR10(
            root=self.root, train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = datasets.CIFAR10(
            root=self.root, train=False, download=True, transform=self.test_transform
        )

        self._init_classnames()

    def _init_classnames(self):
        # CIFAR-10 class names are fixed and provided by the dataset
        self.classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        # No need for a custom mapping like EuroSAT, as CIFAR-10 names are straightforward

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)

    # Optional: Add validation split if needed
    def val_dataloader(self):
        # CIFAR-10 doesn't have a separate validation set by default,
        # but you could split the training set here if desired.
        # For simplicity, reusing test set here (not ideal for real scenarios).
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)