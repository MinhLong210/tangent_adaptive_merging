import os
import re
from typing import List

import lightning.pytorch as pl
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def _pretify_classname(classname):
    l: List[str] = re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", classname)
    l = [i.lower() for i in l]
    out = " ".join(l)
    if out.endswith("al"):
        return out + " area"
    return out


class EuroSATDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        train_transform=None,
        test_transform=None,
    ):
        R"""
        Dataset layout:

            ```
            root/EuroSAT_splits/train
            root/EuroSAT_splits/validation
            root/EuroSAT_splits/test
            ```

        Args:
            root (str): _description_
            train_transform (_type_): _description_
            test_transform (_type_): _description_
            batch_size (int): _description_
            num_workers (int): _description_
            pin_meory (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        self.train_dir = os.path.join(root, "eurosat", "train")
        # self.val_dir = os.path.join(root, "EuroSAT_splits", "validation")
        self.test_dir = os.path.join(root, "eurosat", "test")

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.train_dataset = datasets.ImageFolder(
            self.train_dir, transform=self.train_transform
        )
        # self.val_dataset = datasets.ImageFolder(
        #     self.val_dir, transform=self.test_transform
        # )
        self.test_dataset = datasets.ImageFolder(
            self.test_dir, transform=self.test_transform
        )

        self._init_classnames()

    def _init_classnames(self):
        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classes = [
            idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))
        ]
        # self.classes = [_pretify_classname(c) for c in self.classes]
        ours_to_open_ai = {
            "AnnualCrop": "Annual",  # Assuming "Annula" is a typo
            "Forest": "forest",
            "HerbaceousVegetation": "brushland or shrubland",
            "Highway": "highway or road",
            "Industrial": "industrial buildings or commercial buildings",
            "Pasture": "pasture land",
            "PermanentCrop": "permanent crop land",
            "Residential": "residential buildings or homes or apartments",
            "River": "river",
            "SeaLake": "lake or sea"
        }
        for i in range(len(self.classes)):
            self.classes[i] = ours_to_open_ai[self.classes[i]]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
