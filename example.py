import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class ParkinsonVoiceDatasetLeaky(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
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


from typing import Dict

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import (CIFAR10, KMNIST, MNIST, SVHN, USPS,
                                  VisionDataset)

from learning_to_score.datasets.datasets import (
    Normalize, ParkinsonVoiceDataset, SpiralParkinsonsDrawingsDataset,
    WaveParkinsonsDrawingsDataset)

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
                download=False,
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
        elif (
            self.dataset_obj is ParkinsonVoiceDataset
            or self.dataset_obj is ParkinsonVoiceDatasetLeaky
        ):
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
            while self.dataset[idx_p][2] != self.dataset[idx][2]:
                idx_p = np.random.randint(self.__len__())

            idx_n = np.random.randint(self.__len__())
            while self.dataset[idx_n][2] == self.dataset[idx][2]:
                idx_n = np.random.randint(self.__len__())

            try:
                x_a, y_a, side_information_a = self.dataset[idx]
                x_p, y_p, side_information_p = self.dataset[idx_p]
                x_n, y_n, side_information_n = self.dataset[idx_n]

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
                x_a, y_a, side_information_a = self.dataset[idx]
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


from learning_to_score.datasets.datasets import DataModule
from learning_to_score.model import (Model, VoiceDecoder,
                                                     VoiceEncoder)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    num_epochs=30,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            (
                x_a,
                x_p,
                x_n,
                side_information_a,
                side_information_p,
                side_information_n,
                y_a,
                y_p,
                y_n,
            ) = [t.to(device) for t in batch]

            optimizer.zero_grad()

            (
                encoded_x_a,
                encoded_x_p,
                encoded_x_n,
                z_a,
                z_p,
                z_n,
                reconstructed_x_a,
                reconstructed_x_p,
                reconstructed_x_n,
                mu_x_a,
                mu_x_p,
                mu_x_n,
                log_sigma2_x_a,
                log_sigma2_x_p,
                log_sigma2_x_n,
                logit_a,
                logit_p,
                logit_n,
                side_information_logit_a,
                side_information_logit_p,
                side_information_logit_n,
            ) = model(x_a, x_p, x_n)

            loss = model.loss_function(
                x_a,
                x_p,
                x_n,
                reconstructed_x_a,
                reconstructed_x_p,
                reconstructed_x_n,
                encoded_x_a,
                encoded_x_p,
                encoded_x_n,
                z_a,
                z_p,
                z_n,
                mu_x_a,
                mu_x_p,
                mu_x_n,
                log_sigma2_x_a,
                log_sigma2_x_p,
                log_sigma2_x_n,
                logit_a,
                logit_p,
                logit_n,
                side_information_logit_a,
                side_information_logit_p,
                side_information_logit_n,
                side_information_a,
                side_information_p,
                side_information_n,
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

        # # Validation
        # model.eval()
        # with torch.no_grad():
        #     val_loss = 0
        #     for batch in val_dataloader:
        #         (
        #             x_a,
        #             side_information_a,
        #             y_a,
        #         ) = [t.to(device) for t in batch]

        #         (
        #             encoded_x_a,
        #             z_a,
        #             reconstructed_x_a,
        #             mu_x_a,
        #             log_sigma2_x_a,
        #             logit_a,
        #             side_information_logit_a,
        #         ) = model(x_a, None, None)

        #     val_loss /= len(val_dataloader)
        #     print(f"Validation Loss: {val_loss:.4f}")

def main():
    config_dict = {
        "alpha": 10,
        "beta": 0.01,
        "gamma": 10,
        "delta": 10,
        "zeta": 1,
        "eta": 0.4,
        "latent_dim": 25,
        "enc_out_dim": 10,
        "max_epochs": 100,
        "n_clusters": 1,
        "triplet_loss_margin": 1.0,
        "flatten": False,
        "side_information_type": "pure",
        "side_info_dim": 1,
        "dataset": "ParkinsonVoiceDatasetLeaky",  # ParkinsonsVoiceDataset / MNIST / SVHN / USPS / KMNIST / CIFAR10 / WaveParkinsonsDrawingsDataset / SpiralParkinsonsDrawingsDataset / ParkinsonsVoiceDataset
        "side_info_loss_type": "binary_cross_entropy",  # "cross_entropy"
    }
    config = AttrDict(config_dict)


    # Create the model
    model = Model(
        encoder=VoiceEncoder,
        decoder=VoiceDecoder,
        enc_out_dim=config.enc_out_dim,
        latent_dim=config.latent_dim,
        n_clusters=config.n_clusters,
        side_info_dim=config.side_info_dim,
        side_info_loss_type=config.side_info_loss_type,
        triplet_loss_margin=config.triplet_loss_margin,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        delta=config.delta,
        zeta=config.zeta,
        eta=config.eta,
    )


    dataloader = get_triplet_dataset(
            ParkinsonVoiceDatasetLeaky,
            ".",
            128,
            config.side_information_type,
            flatten=config.flatten,
        )

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    train(model, dataloader.train_dataloader(), dataloader.val_dataloader(), optimizer, num_epochs=config.max_epochs)

if __name__ == '__main__':
    main()