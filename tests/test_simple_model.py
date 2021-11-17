import unittest

import pytorch_lightning as pl
import torch

from learning_to_score.datasets import ClustersDataset, DataModule
from learning_to_score.model import Model


class TestScorer(unittest.TestCase):
    def setUp(self) -> None:
        pl.seed_everything(0)
        self.model = Model(
            input_dim=10, latent_dim=3, encoder_hidden_dims=[5], triplet_loss_margin=1.0
        )

    def test_forward(self):
        x_a = torch.randn(10)
        x_p = x_a * 1.1
        x_n = x_a * 2

        (
            encoded_x_a,
            encoded_x_p,
            encoded_x_n,
            reconstructed_x_a,
            reconstructed_x_p,
            reconstructed_x_n,
        ) = self.model(x_a, x_p, x_n)

    def test_loss(self):
        x_a = torch.randn(10)
        x_p = x_a * 1.1
        x_n = x_a * 2

        side_information_a = torch.ones(1)
        side_information_p = side_information_a * 1.1
        side_information_n = side_information_a * 2

        (
            encoded_x_a,
            encoded_x_p,
            encoded_x_n,
            reconstructed_x_a,
            reconstructed_x_p,
            reconstructed_x_n,
        ) = self.model(x_a, x_p, x_n)

        score_x_a = self.model.get_score(x_a)
        score_x_p = self.model.get_score(x_p)
        score_x_n = self.model.get_score(x_n)

        loss = self.model.loss_function(
            x_a,
            x_p,
            x_n,
            reconstructed_x_a,
            reconstructed_x_p,
            reconstructed_x_n,
            encoded_x_a,
            encoded_x_p,
            encoded_x_n,
            score_x_a,
            score_x_p,
            score_x_n,
            side_information_a,
            side_information_p,
            side_information_n,
        )
        print(loss)

    def test_training_step(self):
        x_a = torch.randn(10)
        x_p = x_a * 1.1
        x_n = x_a * 2

        side_information_a = torch.ones(1)
        side_information_p = side_information_a * 1.1
        side_information_n = side_information_a * 2

        batch = (
            x_a,
            x_p,
            x_n,
            side_information_a,
            side_information_p,
            side_information_n,
        )

        loss = self.model.training_step(batch, 1)

        print(loss)

    def test_training(self):
        from pytorch_lightning.loggers import TensorBoardLogger

        tb_logger = TensorBoardLogger("tb_logs")

        from pytorch_lightning.loggers import WandbLogger

        wandb_logger = WandbLogger()

        pl.seed_everything(0)

        input_dim = 200
        num_of_clusters = 10
        number_of_samples_in_cluster = 1000
        cluster_dim_mean = 3
        mean = 0
        variance = 1

        triplet_loss_margin = 1
        alpha = 1
        beta = 1
        gamma = 1

        encoder_hidden_dims = [100, 50]
        latent_dim = 20
        epochs = 10

        model = Model(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            triplet_loss_margin=triplet_loss_margin,
            number_of_clusters=num_of_clusters,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        trainer = pl.Trainer(max_epochs=epochs, logger=wandb_logger)

        dataset = ClustersDataset(
            num_of_clusters=num_of_clusters,
            number_of_samples_in_cluster=number_of_samples_in_cluster,
            dim=input_dim,
            cluster_dim_mean=cluster_dim_mean,
            mean=mean,
            variance=variance,
        )
        dataset.setup()

        data_loader = DataModule(dataset)

        trainer.fit(model, data_loader)


if __name__ == "__main__":
    unittest.main()
