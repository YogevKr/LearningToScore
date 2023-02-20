import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Callable, Optional
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(16, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logit = self.encoder(x)
        loss = F.cross_entropy(logit, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logit = self.encoder(x)
        loss = F.cross_entropy(logit, y)
        self.log("train_loss", loss)

class ParkinsonsVoiceDataset(Dataset):
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
        self.filename = "parkinsons_updrs.csv"

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



# data
train_dataset = ParkinsonsVoiceDataset(
                root=".",
                train=True,
                download=True,
            )
val_dataset = ParkinsonsVoiceDataset(
                root=".",
                train=False,
                download=True,
            )

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

from pytorch_lightning.loggers import WandbLogger
import wandb

with wandb.init(
    project="Predicting Severity Of Parkinson", 
    save_code=True,
):
    
    model = Net()

    wandb_logger = WandbLogger()
    wandb.watch(
                model,
                log_freq=10,
                log="all",
                log_graph=True,
            )

    trainer = pl.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        accelerator='gpu',
        devices=1,
        detect_anomaly=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )

    trainer.fit(model, train_loader, val_loader)
