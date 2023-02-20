import glob
import os
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import psutil
import pytorch_lightning as pl
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import (CIFAR10, KMNIST, MNIST, SVHN, USPS,
                                  VisionDataset)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ParkinsonsDrawingsDataset(VisionDataset):

    classes = [
        "0 - healthy",
        "1 - parkinson",
    ]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        data_type=None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.img_dir = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data_type = data_type

        self.data, self.targets = self._load_data()

    def _load_data(self):

        image_dir = os.path.join(
            self.root, self.data_type, "training" if self.train else "testing"
        )

        healthy_file_list = glob.glob(os.path.join(image_dir, "healthy", "*.png"))

        healthy_data = [Image.open(fname).convert("RGB") for fname in healthy_file_list]

        healthy_labels = [0] * len(healthy_data)

        parkinson_file_list = glob.glob(os.path.join(image_dir, "parkinson", "*.png"))

        parkinson_data = [
            Image.open(fname).convert("RGB") for fname in parkinson_file_list
        ]

        parkinson_labels = [1] * len(parkinson_data)

        data = healthy_data + parkinson_data
        targets = healthy_labels + parkinson_labels

        return data, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class SpiralParkinsonsDrawingsDataset(ParkinsonsDrawingsDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            data_type="spiral",
        )


class WaveParkinsonsDrawingsDataset(ParkinsonsDrawingsDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            data_type="wave",
        )

class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        min = sample.min()
        max = sample.max()

        return (sample - min) / (max - min)


class TripletsDataset(Dataset):
    def __init__(
        self,
        dataset_obj,
        data_dir="./data/",
        flatten=False,
        train=True,
        side_information_type="pure",
    ):

        self.data_dir = data_dir
        self.dataset = None
        self.flatten = flatten
        self.train = train
        self.side_information_type = side_information_type
        self.dataset_obj = dataset_obj

    def setup(self):
        transformers = [transforms.ToTensor()]

        if self.dataset_obj is MNIST or self.dataset_obj is KMNIST:

            transformers.append(Normalize())
            if self.flatten:
                transformers.append(transforms.Lambda(lambda x: torch.flatten(x)))

            self.dataset = self.dataset_obj(
                root=self.data_dir,
                train=self.train,
                download=True,
                transform=transforms.Compose(transformers),
            )
        elif self.dataset_obj is SVHN:
            t = "train" if self.train else "test"
            transformers.extend([transforms.Grayscale(), transforms.Resize((28, 28))])

            if self.flatten:
                transformers.append(transforms.Lambda(lambda x: torch.flatten(x)))

            self.dataset = self.dataset_obj(
                root=self.data_dir,
                split=t,
                download=True,
                transform=transforms.Compose(transformers),
            )
        elif self.dataset_obj is USPS:
            if self.flatten:
                transformers.append(transforms.Lambda(lambda x: torch.flatten(x)))

            self.dataset = self.dataset_obj(
                root=self.data_dir,
                train=self.train,
                download=True,
                transform=transforms.Compose(transformers),
            )
        elif self.dataset_obj is CIFAR10:

            transformers.extend([transforms.Grayscale(), transforms.Resize((28, 28))])
            if self.flatten:
                transformers.append(transforms.Lambda(lambda x: torch.flatten(x)))

            self.dataset = self.dataset_obj(
                root=self.data_dir,
                train=self.train,
                download=True,
                transform=transforms.Compose(transformers),
            )
        elif (
            self.dataset_obj is WaveParkinsonsDrawingsDataset
            or self.dataset_obj is SpiralParkinsonsDrawingsDataset
        ):

            transformers.extend([transforms.Grayscale(), transforms.Resize((28, 28))])
            if self.flatten:
                transformers.append(transforms.Lambda(lambda x: torch.flatten(x)))

            self.dataset = self.dataset_obj(
                root=self.data_dir,
                train=self.train,
                download=True,
                transform=transforms.Compose(transformers),
            )
        elif self.dataset_obj is ParkinsonVoiceDataset:

            self.dataset = self.dataset_obj(
                root=self.data_dir,
                train=self.train,
                download=True,
            )
        else:
            raise TypeError

    def __len__(self):
        return len(self.dataset)

    def _get_side_information(self, y_a, y_p=None, y_n=None):
        if self.side_information_type == "pure":
            if self.train:
                return (y_a, y_p, y_n)
            else:
                return y_a
        elif self.side_information_type == "mod_2":
            if self.train:
                return (y_a // 2, y_p // 2, y_n // 2)
            else:
                return y_a // 2
        elif self.side_information_type == "unbalanced":
            mapping_dict = {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 1,
                5: 1,
                6: 1,
                7: 2,
                8: 2,
                9: 3,
            }

            if self.train:
                return (
                    mapping_dict[y_a],
                    mapping_dict[y_p],
                    mapping_dict[y_n],
                )
            else:
                return mapping_dict[y_a]

    def __getitem__(self, idx):

        if self.train:

            idx_p = np.random.randint(self.__len__())
            while self.dataset[idx_p][3] != self.dataset[idx][2]:
                idx_p = np.random.randint(self.__len__())

            idx_n = np.random.randint(self.__len__())
            while self.dataset[idx_n][3] == self.dataset[idx][2]:
                idx_n = np.random.randint(self.__len__())

            try:
                x_a, y_a, side_information_a, _ = self.dataset[idx]
                x_p, y_p, side_information_p, _ = self.dataset[idx_p]
                x_n, y_n, side_information_n, _ = self.dataset[idx_n]

            except ValueError:
                x_a, y_a = self.dataset[idx]
                x_p, y_p = self.dataset[idx_p]
                x_n, y_n = self.dataset[idx_n]

                (
                    side_information_a,
                    side_information_p,
                    side_information_n,
                ) = self._get_side_information(y_a, y_p, y_n)

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
        else:
            try:
                x_a, y_a, side_information_a, _ = self.dataset[idx]
            except ValueError:
                x_a, y_a = self.dataset[idx]
                side_information_a = self._get_side_information(y_a)

            return (
                x_a,
                side_information_a,
                y_a,
            )

    def get_summery(self) -> Dict:
        return dict(
            dataset_type=type(self).__name__,
        )


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int = 128,
        num_workers=2,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def get_summery(self) -> Dict:
        return self.train_dataset.get_summery()


def get_triplet_dataset(
    dataset_obj,
    data_dir="./data/",
    batch_size=128,
    side_information_type=None,
    flatten=False,
):
    triplets_dataset_train = TripletsDataset(
        dataset_obj,
        data_dir=data_dir,
        side_information_type=side_information_type,
        flatten=flatten,
    )
    triplets_dataset_train.setup()

    triplets_dataset_test = TripletsDataset(
        dataset_obj,
        data_dir=data_dir,
        train=False,
        side_information_type=side_information_type,
        flatten=flatten,
    )
    triplets_dataset_test.setup()

    return DataModule(
        train_dataset=triplets_dataset_train,
        test_dataset=triplets_dataset_test,
        batch_size=batch_size,
        num_workers=psutil.cpu_count(),
    )


class ParkinsonVoiceDataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__()

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.filename = "parkinson_updrs.csv"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets, self.side_info, self.subject_ids = self._load_data()

    def download(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"

        if not os.path.exists(self.raw_folder):
            os.makedirs(self.raw_folder)

        df = pd.read_csv(url)
        df.to_csv(os.path.join(self.raw_folder, self.filename))

    def _check_exists(self) -> bool:
        return os.path.isfile(os.path.join(self.raw_folder, self.filename))

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _load_data(self):

        feature_columns = [
            "Jitter(%)",
            "Jitter(Abs)",
            "Jitter:RAP",
            "Jitter:PPQ5",
            "Jitter:DDP",
            "Shimmer",
            "Shimmer(dB)",
            "Shimmer:APQ3",
            "Shimmer:APQ5",
            "Shimmer:APQ11",
            "Shimmer:DDA",
            "NHR",
            "HNR",
            "RPDE",
            "DFA",
            "PPE",
            "age",
            "sex"
        ]

        target_column = "total_UPDRS_bool"
        side_info_column = "total_UPDRS_bucket"  #  "total_UPDRS_bucket" / "total_UPDRS_int" / "total_UPDRS_bool" / "motor_UPDRS" / "total_UPDRS"

        test_users_ids = [3, 6, 15, 17, 24, 26, 28, 30, 32, 41]

        df_ = pd.read_csv(os.path.join(self.raw_folder, self.filename))

        df_["total_UPDRS_bool"] = (df_["total_UPDRS"] > 25).astype(int)
        df_["total_UPDRS_int"] = df_["total_UPDRS"].astype(int)
        df_["total_UPDRS_bucket"] = (df_["total_UPDRS"] // 5).astype(int)

        df_train = df_.loc[~df_["subject#"].isin(test_users_ids)]
        len(set(df_train["subject#"].unique()).intersection(set(test_users_ids))) == 0

        df_test = df_.loc[df_["subject#"].isin(test_users_ids)]
        len(set(df_test["subject#"].unique()).intersection(set(test_users_ids))) == len(
            test_users_ids
        )

        train_X = df_train[feature_columns].values
        min_max_scaler = self.fit_min_max_scaler(train_X)

        df = df_train if self.train else df_test

        data = min_max_scaler.transform(df[feature_columns].values)
        targets = df[target_column].values
        side_info = df[side_info_column].values
        subject_id = df[side_info_column].values

        return data, targets, side_info, subject_id

    @staticmethod
    def fit_min_max_scaler(train_X):
        scaler = MinMaxScaler()
        scaler.fit(train_X)
        return scaler

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        instance = self.data[idx]
        label = self.targets[idx]
        side_info = self.side_info[idx]
        subject_id = self.subject_ids[idx]
        return instance.astype(np.float32), label, side_info, subject_id


class ParkinsonVoiceDatasetLeaky(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__()

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.filename = "parkinson_updrs.csv"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets, self.side_info = self._load_data()

    def download(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"

        if not os.path.exists(self.raw_folder):
            os.makedirs(self.raw_folder)

        df = pd.read_csv(url)
        df.to_csv(os.path.join(self.raw_folder, self.filename))

    def _check_exists(self) -> bool:
        return os.path.isfile(os.path.join(self.raw_folder, self.filename))

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _load_data(self):

        feature_columns = [
            "Jitter(%)",
            "Jitter(Abs)",
            "Jitter:RAP",
            "Jitter:PPQ5",
            "Jitter:DDP",
            "Shimmer",
            "Shimmer(dB)",
            "Shimmer:APQ3",
            "Shimmer:APQ5",
            "Shimmer:APQ11",
            "Shimmer:DDA",
            "NHR",
            "HNR",
            "RPDE",
            "DFA",
            "PPE",
            "age",
            "sex",
        ]

        target_column = "total_UPDRS_bool"
        side_info_column = "total_UPDRS_bool"  #  "total_UPDRS_bucket" / "total_UPDRS_int" / "total_UPDRS_bool" / "motor_UPDRS" / "total_UPDRS"

        df_ = pd.read_csv(os.path.join(self.raw_folder, self.filename))

        df_["total_UPDRS_bool"] = (df_["total_UPDRS"] > 25).astype(int)
        df_["total_UPDRS_int"] = df_["total_UPDRS"].astype(int)
        df_["total_UPDRS_bucket"] = (df_["total_UPDRS"] // 10).astype(int)
        
        scaler = MinMaxScaler()
        scaler.fit(df_[feature_columns])
        
        df_train, df_test = train_test_split(df_, test_size=0.2, random_state=42)

        if self.train:
            data = scaler.transform(df_train[feature_columns])
            targets = df_train[target_column].values
            side_info = df_train[side_info_column].values
        else:
            data = scaler.transform(df_test[feature_columns])
            targets = df_test[target_column].values
            side_info = df_test[side_info_column].values

        return data, targets, side_info

    def __len__(self):
        return len(self.targets)
	
    def __getitem__(self, idx):	
        instance = self.data[idx]	
        label = self.targets[idx]	
        side_info = self.side_info[idx]	
        return instance.astype(np.float32), label, side_info

class ParkinsonVoiceDataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__()

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.filename = "parkinson_updrs.csv"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets, self.side_info, self.subject_ids = self._load_data()

    def download(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"

        if not os.path.exists(self.raw_folder):
            os.makedirs(self.raw_folder)

        df = pd.read_csv(url)
        df.to_csv(os.path.join(self.raw_folder, self.filename))

    def _check_exists(self) -> bool:
        return os.path.isfile(os.path.join(self.raw_folder, self.filename))

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _load_data(self):

        feature_columns = [
            "Jitter(%)",
            "Jitter(Abs)",
            "Jitter:RAP",
            "Jitter:PPQ5",
            "Jitter:DDP",
            "Shimmer",
            "Shimmer(dB)",
            "Shimmer:APQ3",
            "Shimmer:APQ5",
            "Shimmer:APQ11",
            "Shimmer:DDA",
            "NHR",
            "HNR",
            "RPDE",
            "DFA",
            "PPE",
            "age",
            "sex"
        ]

        target_column = "total_UPDRS_bool"
        side_info_column = "total_UPDRS_bucket"  #  "total_UPDRS_bucket" / "total_UPDRS_int" / "total_UPDRS_bool" / "motor_UPDRS" / "total_UPDRS"

        test_users_ids = [3, 6, 15, 17, 24, 26, 28, 30, 32, 41]

        df_ = pd.read_csv(os.path.join(self.raw_folder, self.filename))

        df_["total_UPDRS_bool"] = (df_["total_UPDRS"] > 25).astype(int)
        df_["total_UPDRS_int"] = df_["total_UPDRS"].astype(int)
        df_["total_UPDRS_bucket"] = (df_["total_UPDRS"] // 5).astype(int)

        df_train = df_.loc[~df_["subject#"].isin(test_users_ids)]
        len(set(df_train["subject#"].unique()).intersection(set(test_users_ids))) == 0

        df_test = df_.loc[df_["subject#"].isin(test_users_ids)]
        len(set(df_test["subject#"].unique()).intersection(set(test_users_ids))) == len(
            test_users_ids
        )

        train_X = df_train[feature_columns].values
        min_max_scaler = self.fit_min_max_scaler(train_X)

        df = df_train if self.train else df_test

        data = min_max_scaler.transform(df[feature_columns].values)
        targets = df[target_column].values
        side_info = df[side_info_column].values
        subject_id = df[side_info_column].values

        return data, targets, side_info, subject_id

    @staticmethod
    def fit_min_max_scaler(train_X):
        scaler = MinMaxScaler()
        scaler.fit(train_X)
        return scaler

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        instance = self.data[idx]
        label = self.targets[idx]
        side_info = self.side_info[idx]
        subject_id = self.subject_ids[idx]
        return instance.astype(np.float32), label, side_info, subject_id


class ParkinsonVoiceDatasetLeaky(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__()

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.filename = "parkinson_updrs.csv"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets, self.side_info = self._load_data()

    def download(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"

        if not os.path.exists(self.raw_folder):
            os.makedirs(self.raw_folder)

        df = pd.read_csv(url)
        df.to_csv(os.path.join(self.raw_folder, self.filename))

    def _check_exists(self) -> bool:
        return os.path.isfile(os.path.join(self.raw_folder, self.filename))

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _load_data(self):

        feature_columns = [
            "Jitter(%)",
            "Jitter(Abs)",
            "Jitter:RAP",
            "Jitter:PPQ5",
            "Jitter:DDP",
            "Shimmer",
            "Shimmer(dB)",
            "Shimmer:APQ3",
            "Shimmer:APQ5",
            "Shimmer:APQ11",
            "Shimmer:DDA",
            "NHR",
            "HNR",
            "RPDE",
            "DFA",
            "PPE",
            "age",
            "sex",
        ]

        target_column = "total_UPDRS_bool"
        side_info_column = "total_UPDRS_bool"  #  "total_UPDRS_bucket" / "total_UPDRS_int" / "total_UPDRS_bool" / "motor_UPDRS" / "total_UPDRS"

        df_ = pd.read_csv(os.path.join(self.raw_folder, self.filename))

        df_["total_UPDRS_bool"] = (df_["total_UPDRS"] > 25).astype(int)
        df_["total_UPDRS_int"] = df_["total_UPDRS"].astype(int)
        df_["total_UPDRS_bucket"] = (df_["total_UPDRS"] // 10).astype(int)
        
        scaler = MinMaxScaler()
        scaler.fit(df_[feature_columns])
        
        df_train, df_test = train_test_split(df_, test_size=0.2, random_state=42)

        if self.train:
            data = scaler.transform(df_train[feature_columns])
            targets = df_train[target_column].values
            side_info = df_train[side_info_column].values
        else:
            data = scaler.transform(df_test[feature_columns])
            targets = df_test[target_column].values
            side_info = df_test[side_info_column].values

        return data, targets, side_info

    def __len__(self):
        return len(self.targets)
	
    def __getitem__(self, idx):	
        instance = self.data[idx]	
        label = self.targets[idx]	
        side_info = self.side_info[idx]	
        return instance.astype(np.float32), label, side_info

datasets_dict = {
    "MNIST": MNIST,
    "SVHN": SVHN,
    "USPS": USPS,
    "KMNIST": KMNIST,
    "CIFAR10": CIFAR10,
    "WaveParkinsonsDrawingsDataset": WaveParkinsonsDrawingsDataset,
    "SpiralParkinsonsDrawingsDataset": SpiralParkinsonsDrawingsDataset,
    "ParkinsonsVoiceDataset": ParkinsonVoiceDataset
}