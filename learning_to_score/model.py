import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import matplotlib.pyplot as plt
import wandb
import umap
import umap.plot


class CNNEncoder(nn.Module):
    def __init__(
        self,
    ):
        super(CNNEncoder, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=4,
                stride=2,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.linear = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x_encoded = self.linear(x)
        return x_encoded


class CNNDecoder(nn.Module):
    def __init__(self, input_dim=20):
        super(CNNDecoder, self).__init__()

        a = 128
        b = 2

        self.decoder = nn.Sequential(
            nn.Linear(input_dim + 10, a * b * b),
            nn.Unflatten(1, (a, b, b)),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=4,
                stride=2,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x_pro = self.decoder(z)
        return x_pro


class ToynetEncoder(nn.Module):
    def __init__(
        self,
    ):
        super(ToynetEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        return x_encoded


class ToynetDecoder(nn.Module):
    def __init__(self, input_dim=20):
        super(ToynetDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 784),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x_pro = self.decoder(z)
        return x_pro


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = int(max(Y_pred.max(), Y.max()) + 1)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], int(Y[i])] += 1
    ind = list(zip(*linear_assignment(w.max() - w)))
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, ind


class Scorer(nn.Module):
    def __init__(
        self,
        latent_dim,
        number_of_clusters,
    ):
        super(Scorer, self).__init__()
        self.latent_dim = latent_dim
        self.number_of_clusters = number_of_clusters

        self.scorer = nn.Sequential(
            nn.Linear(latent_dim, 500),
            nn.ReLU(True),
            nn.Linear(500, number_of_clusters),
        )

    def forward(self, x):
        x = self.scorer(x)
        return x


class Model(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        enc_out_dim=1024,
        latent_dim=2,
        n_clusters=10,
        alpha=1,
        beta=1,
        zeta=1,
        eta=0.5,
        delta=10,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.zeta = zeta
        self.eta = eta
        self.delta = delta

        self.encoder = encoder()
        self.decoder = decoder(input_dim=latent_dim)

        self.fc_mu_l = nn.Linear(enc_out_dim, latent_dim)
        self.fc_log_sigma2_l = nn.Linear(enc_out_dim, latent_dim)

        self.scorer = Scorer(latent_dim, n_clusters)

    def on_train_start(self):
        self.hparams.update(self.trainer.datamodule.get_summery())
        self.logger.log_hyperparams(self.hparams)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    @staticmethod
    def sample(mu, log_sigma2):
        std = torch.exp(log_sigma2 / 2)
        z = torch.randn_like(mu) * std + mu

        return z

    @staticmethod
    def vae_kl(mu, log_var):
        return torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

    def forward(self, x, y=None):

        encoded_x = self.encoder(x)

        mu, log_sigma2 = self.fc_mu_l(encoded_x), self.fc_log_sigma2_l(encoded_x)

        z = self.sample(mu, log_sigma2)

        logit = self.scorer(z)

        reconstructed_x = self.decoder(torch.cat([z, logit], dim=1))

        return (
            encoded_x,
            z,
            reconstructed_x,
            mu,
            log_sigma2,
            logit,
        )

    def loss_function(
        self, x, y, reconstructed_x, encoded_x, z, mu, log_sigma2, logit, labeled_logit
    ):

        reconstruction_loss = self.alpha * F.binary_cross_entropy(reconstructed_x, x)

        kl_loss = self.beta * self.vae_kl(mu, log_sigma2)

        def conditinal_entropy(logit):
            probs = F.softmax(logit, dim=1)
            return -torch.mean(torch.sum(probs * (torch.log(probs + 1e-8)), 1))

        score_given_z_entropy = conditinal_entropy(logit)

        def score_entropy(logit):
            probs = F.softmax(logit, dim=1)
            score_probs = torch.mean(probs, 0)
            return -torch.sum(score_probs * torch.log(score_probs))

        score_entropy = score_entropy(logit)
        mutual_info = self.zeta * (
            self.eta * score_entropy - (1 - self.eta) * score_given_z_entropy
        )

        if labeled_logit is None and y is None:
            cross_entropy_loss = 0
        else:
            cross_entropy_loss = self.delta * F.cross_entropy(labeled_logit, y)

        loss = reconstruction_loss + kl_loss - mutual_info + cross_entropy_loss

        self.log_dict(
            {
                "loss": loss,
                "recon_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                "mutual_info": mutual_info,
                "score_entropy": score_entropy,
                "score_given_z_entropy": score_given_z_entropy,
                "cross_entropy": cross_entropy_loss,
            }
        )

        return loss

    def training_step(self, batch, batch_idx):
        unlabeled_data, labeled_data = batch

        if unlabeled_data and labeled_data:
            unlabeled_x, _ = unlabeled_data
            labeled_x, labeled_y = labeled_data

            (
                unlabeled_encoded_x,
                unlabeled_z,
                unlabeled_reconstructed_x,
                unlabeled_mu,
                unlabeled_log_sigma2,
                unlabeled_logit,
            ) = self(unlabeled_x)

            (
                labeled_encoded_x,
                labeled_z,
                labeled_reconstructed_x,
                labeled_mu,
                labeled_log_sigma2,
                labeled_logit,
            ) = self(labeled_x)

            x = torch.cat([unlabeled_x, labeled_x], 0)

            reconstructed_x = torch.cat(
                [unlabeled_reconstructed_x, labeled_reconstructed_x], 0
            )

            encoded_x = torch.cat([unlabeled_encoded_x, labeled_encoded_x], 0)

            z = torch.cat([unlabeled_z, labeled_z], 0)

            mu = torch.cat([unlabeled_mu, labeled_mu], 0)

            log_sigma2 = torch.cat([unlabeled_log_sigma2, labeled_log_sigma2], 0)

            logit = torch.cat([unlabeled_logit, labeled_logit], 0)

            loss = self.loss_function(
                x,
                labeled_y,
                reconstructed_x,
                encoded_x,
                z,
                mu,
                log_sigma2,
                logit,
                labeled_logit,
            )
        elif not unlabeled_data:
            x, y = labeled_data

            (
                encoded_x,
                z,
                reconstructed_x,
                mu,
                log_sigma2,
                logit,
            ) = self(x)

            loss = self.loss_function(
                x, y, reconstructed_x, encoded_x, z, mu, log_sigma2, logit, logit
            )
        elif not labeled_data:
            x, _ = unlabeled_data
            (
                encoded_x,
                z,
                reconstructed_x,
                mu,
                log_sigma2,
                logit,
            ) = self(x)

            loss = self.loss_function(
                x, None, reconstructed_x, encoded_x, z, mu, log_sigma2, logit, None
            )

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch

        (
            encoded_x,
            z,
            reconstructed_x,
            mu,
            log_sigma2,
            logit,
        ) = self(x)

        return torch.argmax(logit, axis=1)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        (
            encoded_x,
            z,
            reconstructed_x,
            mu,
            log_sigma2,
            logit,
        ) = self(x)

        pred = self.predict_step(batch, batch_idx)

        return mu, log_sigma2, pred, y, reconstructed_x

    def validation_epoch_end(self, outputs):
        mu_s = []
        log_sigma2_s = []
        preds = []
        y_as = []
        reconstructed_xs = []

        for out in outputs:
            mu, log_sigma2, pred, y_a, reconstructed_x = out
            mu_s.append(mu)
            log_sigma2_s.append(log_sigma2)
            preds.append(pred)
            y_as.append(y_a)
            reconstructed_xs.append(reconstructed_x)

        mu = torch.cat(mu_s).cpu().detach().numpy()
        log_sigma2_x_a = torch.cat(log_sigma2_s).cpu().detach()
        preds = torch.cat(preds).cpu().numpy()
        y_a = torch.cat(y_as).cpu().numpy()
        reconstructed_xs = torch.cat(reconstructed_xs).cpu().numpy()

        def plot(mu_x_a, labels):
            fig, ax = plt.subplots()
            scatter = ax.scatter(
                mu_x_a[:, 0], mu_x_a[:, 1], c=np.squeeze(labels), alpha=0.4
            )
            ax.legend(
                *scatter.legend_elements(),
            )
            return fig

        if self.latent_dim == 2:
            q_z_x_true_plot = plot(mu, np.squeeze(y_a))
            q_z_x_score_plot = plot(mu, np.squeeze(preds))
        else:
            mapper = umap.UMAP().fit(mu)
            q_z_x_true_plot = umap.plot.points(mapper, labels=np.squeeze(y_a))
            q_z_x_score_plot = umap.plot.points(mapper, labels=np.squeeze(preds))

        def plot_mnist(images, labels):
            num_row = 2
            num_col = 5
            fig, axes = plt.subplots(
                num_row, num_col, figsize=(1.5 * num_col, 2 * num_row)
            )
            for i in range(10):
                img = images[i]
                if img.ndim == 3:
                    img = img[0]
                else:
                    img = img.reshape((28, 28))

                ax = axes[i // num_col, i % num_col]
                ax.imshow(img, cmap="gray")
                ax.set_title("Label: {}".format(labels[i]))
            plt.tight_layout()
            return plt

        self.logger.experiment.log(
            {
                "preds": preds,
                "mu": mu,
                "log_sigma2 norm": torch.linalg.norm(log_sigma2_x_a, axis=1).mean(),
                "y_a": y_a,
            }
        )

        acc, ind = cluster_acc(preds, y_a)

        self.logger.experiment.log(
            {
                "acc": acc,
                "error-rate": (1 - acc) * 100,
                "q(z|x) 2d": wandb.Image(q_z_x_true_plot),
                "q(z|x) 2d score": wandb.Image(q_z_x_score_plot),
                "reconstructed": plot_mnist(reconstructed_xs, y_a),
            }
        )


