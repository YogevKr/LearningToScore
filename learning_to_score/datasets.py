import torch
from torchvision.datasets import MNIST
from torchvision import transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from typing import Dict


class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        min = sample.min()
        max = sample.max()

        return sample - min / (max - min)


class MNISTTrainDataset:
    def __init__(
        self,
        data_dir=".",
        labeled_percentage=0.1,
        num_labeled=None,
        flatten=False,
    ):

        self.data_dir = data_dir
        self.labeled_percentage = labeled_percentage
        self.num_labeled = num_labeled
        self.flatten = flatten

        self.labeled_dataset = None
        self.unlabeled_dataset = None

        self.labeled_size = None
        self.unlabeled_size = None

    def setup(self):
        transformers = [transforms.ToTensor(), Normalize()]

        if self.flatten:
            transformers.append(T.Lambda(lambda x: torch.flatten(x)))

        full_train_dataset = MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transforms.Compose(transformers),
        )

        if self.num_labeled is not None:
            label_size = self.num_labeled
        else:
            label_size = int(len(full_train_dataset) * self.labeled_percentage)

        self.labeled_dataset, self.unlabeled_dataset = torch.utils.data.random_split(
            full_train_dataset, [label_size, len(full_train_dataset) - label_size]
        )

        self.labeled_size = len(self.labeled_dataset)
        self.unlabeled_size = len(self.unlabeled_dataset)

    def get_datasets(self):
        self.setup()
        return self.unlabeled_dataset, self.labeled_dataset


def get_mnist(data_dir=".", train=True, flatten=False):
    transformers = [transforms.ToTensor(), Normalize()]

    if flatten:
        transformers.append(T.Lambda(lambda x: torch.flatten(x)))

    return MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transforms.Compose(transformers),
    )


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        unlabeled_train_dataset: Dataset,
        labeled_train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size=100,
    ):
        super().__init__()
        self.unlabeled_train_dataset = unlabeled_train_dataset
        self.labeled_train_dataset = labeled_train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):

        if len(self.unlabeled_train_dataset) == 0:
            return zip_longest(
                [],
                DataLoader(
                    self.labeled_train_dataset, shuffle=True, batch_size=self.batch_size
                ),
            )
        elif len(self.labeled_train_dataset) == 0:
            return zip_longest(
                DataLoader(
                    self.unlabeled_train_dataset,
                    shuffle=True,
                    batch_size=self.batch_size,
                ),
                [],
            )
        else:
            return zip_longest(
                DataLoader(
                    self.unlabeled_train_dataset,
                    shuffle=True,
                    batch_size=self.batch_size,
                ),
                DataLoader(
                    self.labeled_train_dataset, shuffle=True, batch_size=self.batch_size
                ),
            )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=800)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=800)

    def get_summery(self) -> Dict:
        return {}
