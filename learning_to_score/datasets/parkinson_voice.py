import os
from typing import Callable, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

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
