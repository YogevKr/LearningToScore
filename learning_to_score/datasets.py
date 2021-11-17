from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class ClustersDataset(Dataset):
    def __init__(
        self,
        num_of_clusters=3,
        number_of_samples_in_cluster=1000,
        dim=100,
        cluster_dim_mean=10,
        mean=0,
        variance=2,
        side_information_type="pure",
    ):
        self.num_of_clusters = num_of_clusters
        self.number_of_samples_in_cluster = number_of_samples_in_cluster
        self.dim = dim
        self.cluster_dim_mean = cluster_dim_mean
        self.mean = mean
        self.variance = variance
        self.side_information_type = side_information_type

        self.X = None
        self.S = None
        self.Y = None

    def setup(self):

        self.X = torch.cat(
            [
                torch.cat(
                    (
                        torch.normal(
                            self.mean,
                            self.variance,
                            size=(self.number_of_samples_in_cluster, i),
                        ),
                        torch.normal(
                            self.cluster_dim_mean,
                            self.variance,
                            size=(self.number_of_samples_in_cluster, 1),
                        ),
                        torch.normal(
                            self.mean,
                            self.variance,
                            size=(
                                self.number_of_samples_in_cluster,
                                self.dim - (i + 1),
                            ),
                        ),
                    ),
                    dim=1,
                )
                for i in range(self.num_of_clusters)
            ]
        )

        self.Y = torch.cat(
            [
                torch.unsqueeze(torch.ones(self.number_of_samples_in_cluster) * i, 1)
                for i in range(self.num_of_clusters)
            ]
        )

        self.S = self.get_side_information()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        idx_p = np.random.randint(self.__len__())
        while self.Y[idx_p] != self.Y[idx]:
            idx_p = np.random.randint(self.__len__())

        idx_n = np.random.randint(self.__len__())
        while self.Y[idx_n] == self.Y[idx]:
            idx_n = np.random.randint(self.__len__())

        x_a = self.X[idx]
        x_p = self.X[idx_p]
        x_n = self.X[idx_n]
        side_information_a = self.S[idx]
        side_information_p = self.S[idx_p]
        side_information_n = self.S[idx_n]
        y_a = self.Y[idx]
        y_p = self.Y[idx_p]
        y_n = self.Y[idx_n]

        return (
            x_a,
            x_p,
            x_n,
            side_information_a,
            side_information_p,
            side_information_n,
            y_a,
            y_p,
            y_n,
        )

    def get_summery(self) -> Dict:
        return dict(
            dataset_type=type(self).__name__,
            dataset_num_of_clusters=self.num_of_clusters,
            dataset_number_of_samples_in_cluster=self.number_of_samples_in_cluster,
            dataset_dim=self.dim,
            dataset_cluster_dim_mean=self.cluster_dim_mean,
            dataset_mean=self.mean,
            dataset_variance=self.variance,
            dataset_side_information_type=self.side_information_type,
        )

    def get_side_information(self):
        if self.side_information_type == "pure":
            side_information = self.Y
        elif self.side_information_type == "merged_clusters":
            side_information = torch.cat(
                [
                    torch.unsqueeze(
                        torch.ones(self.number_of_samples_in_cluster * 2) * i, 1
                    )
                    for i in range(self.num_of_clusters // 2)
                ]
            )
        else:
            raise ValueError

        return side_information


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.data = dataset

    def train_dataloader(self):
        return DataLoader(self.data, shuffle=True, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.data, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.data, batch_size=32)

    def get_summery(self) -> Dict:
        return self.data.get_summery()
