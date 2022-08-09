from datasets import get_triplet_dataset, datasets_dict
from model import CNNDecoder, CNNEncoder, Model
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import argparse

import wandb


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        pl.seed_everything(1234)
        
        mnist_dataloader = get_triplet_dataset(
            datasets_dict[config.dataset],
            ".",
            1024, 
            config.side_inforamtion_type,
            flatten=config.flatten
        )

        dl = mnist_dataloader

        model = Model(
            encoder=CNNDecoder,
            decoder=CNNEncoder,
            enc_out_dim=config.enc_out_dim,
            latent_dim=config.latent_dim,
            n_clusters=config.n_clusters,
            side_info_dim=config.side_info_dim,
            triplet_loss_margin=config.triplet_loss_margin,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            delta=config.delta,
            zeta=config.zeta,
            eta=config.eta,
        )
        wandb_logger = WandbLogger()
        wandb.watch(
                model,
                log_freq=10,
                log="all",
                log_graph=True,
            )
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            logger=wandb_logger,
            accelerator='gpu',
            devices=1,
            detect_anomaly=True,
            log_every_n_steps=1,
            num_sanity_val_steps=0
        )

        trainer.fit(model, dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--flatten", type=bool)
    parser.add_argument("--num_labeled", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--zeta", type=float)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--enc_out_dim", type=int)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--n_clusters", type=int)
    parser.add_argument("--triplet_loss_margin", type=float)
    parser.add_argument("--side_inforamtion_type", type=str)
    parser.add_argument("--side_info_dim", type=int)
    parser.add_argument("--dataset", type=str)

    train(parser.parse_args())
