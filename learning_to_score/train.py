from datasets import MNISTTrainDataset, get_mnist, DataModule
from model import CNNDecoder, CNNEncoder, Model
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import argparse

import wandb


def train(config=None):
    with wandb.init(config=config, project="un2full vib"):
        config = wandb.config

        pl.seed_everything(1234)


        unlabeled_dataset, labeled_dataset = MNISTTrainDataset(
            num_labeled=config.num_labeled,
            flatten=config.flatten,
        ).get_datasets()

        data_loader = DataModule(
            unlabeled_train_dataset=unlabeled_dataset,
            labeled_train_dataset=labeled_dataset,
            test_dataset=get_mnist(train=False, flatten=config.flatten),
        )

        max_epochs = 100

        model = Model(
            CNNEncoder,
            CNNDecoder,
            alpha=config.alpha,
            beta=config.beta,
            zeta=config.zeta,
            eta=config.eta,
            delta=config.delta,
            enc_out_dim=config.enc_out_dim,
            latent_dim=config.latent_dim,
        )

        wandb_logger = WandbLogger()
        
        wandb.watch(model)
        wandb.config.update(model.hparams)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=wandb_logger,
            gpus=1,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            reload_dataloaders_every_n_epochs=1,
        )

        trainer.fit(model, data_loader)

        return trainer, model, data_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--flatten", type=bool)
    parser.add_argument("--num_labeled", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--zeta", type=float)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--enc_out_dim", type=int)
    parser.add_argument("--latent_dim", type=int)

    train(parser.parse_args())
