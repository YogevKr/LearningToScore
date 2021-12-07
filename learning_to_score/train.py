import pytorch_lightning as pl
from datasets import ClustersDataset, DataModule
from model import Model
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import argparse

import wandb

pl.seed_everything(0)


def train(config=None):
    with wandb.init(config=config, save_code=True):
        config = wandb.config
        model = Model(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            encoder_hidden_dims=config.encoder_hidden_dims,
            triplet_loss_margin=config.triplet_loss_margin,
            number_of_clusters=config.num_of_clusters,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            scorer_last_activation_function=config.scorer_last_activation_function,
            triplet_loss_objective=config.triplet_loss_objective,
            uncertainty_loss=config.uncertainty_loss,
        )

        wandb_logger = WandbLogger()

        wandb.config.update(model.hparams)

        trainer = pl.Trainer(
            max_epochs=config.epochs, track_grad_norm=2, logger=wandb_logger
        )

        wandb.watch(model)

        dataset = ClustersDataset(
            num_of_clusters=config.num_of_clusters,
            number_of_samples_in_cluster=config.number_of_samples_in_cluster,
            side_information_type=config.side_information_type,
            cluster_dim_mean=config.cluster_dim_mean,
            mean=config.mean,
            variance=config.variance,
            dim=config.input_dim,
        )
        dataset.setup()

        data_loader = DataModule(dataset)

        trainer.fit(model, data_loader)

        return trainer, model, data_loader


        # Vade
        # dl = DataModule(dataset, batch_size=32)
        #
        # vade = VaDE(
        #     alpha=100,
        #     beta=1,
        #     gamma=0,
        #     triplet_loss_margin=1,
        #     enc_out_dim=50,
        #     latent_dim=20
        # )
        # wandb_logger = WandbLogger()
        # wandb.watch(vade)
        # vade.pre_train(dl, pre_epoch=50)
        # trainer = pl.Trainer(
        #     max_epochs=300,
        #     logger=wandb_logger,
        #     gpus=1,
        #     # detect_anomaly=True,
        #     # log_every_n_steps=1
        # )
        # trainer.fit(vade, dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dim", type=int)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--encoder_hidden_dims")
    parser.add_argument("--triplet_loss_margin", type=float)
    parser.add_argument("--number_of_clusters", type=int, default=10)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--scorer_last_activation_function", type=str)
    parser.add_argument("--triplet_loss_objective", type=str)
    parser.add_argument("--uncertainty_loss", type=bool)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num_of_clusters", type=int)
    parser.add_argument("--number_of_samples_in_cluster", type=int)
    parser.add_argument("--side_information_type", type=str)
    parser.add_argument("--cluster_dim_mean", type=float)
    parser.add_argument("--mean", type=float)
    parser.add_argument("--variance", type=float)

    train(parser.parse_args())
