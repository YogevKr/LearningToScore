import math
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from scipy.optimize import linear_sum_assignment as linear_assignment
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score


class LinearLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out

class MnistCNNEncoder(nn.Module):	
    def __init__(	
        self,	
    ):	
        super(MnistCNNEncoder, self).__init__()	
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
        )	
        self.linear = nn.Sequential(	
            nn.Linear(1600, 1000),	
            nn.ReLU(True),	
            nn.Linear(1000, 500),	
            nn.ReLU(True),	
            nn.Linear(500, 100),	
            nn.ReLU(True),	
        )	
    def forward(self, x):	
        x = self.cnn(x)	
        x = x.view(x.size(0), -1)	
        x_encoded = self.linear(x)	
        return x_encoded	

class MnistCNNDecoder(nn.Module):	
    def __init__(self, input_dim=20):	
        super(MnistCNNDecoder, self).__init__()	
        a = 128	
        b = 2	
        self.decoder = nn.Sequential(	
            nn.Linear(input_dim, a * b * b),	
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
        )

        self.linear = nn.Sequential(
            nn.Linear(1600, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 500),
            nn.ReLU(True),
            nn.Linear(500, 100),
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
            nn.Linear(input_dim, a * b * b),
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

class ParkinsonCNNEncoder(nn.Module):
    def __init__(
        self,
    ):
        super(ParkinsonCNNEncoder, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=4,
                stride=2,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=2,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.linear = nn.Sequential(
            nn.Linear(1568, 128),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x_encoded = self.linear(x)
        return x_encoded


class ParkinsonCNNDecoder(nn.Module):
    def __init__(self, input_dim=20):
        super(ParkinsonCNNDecoder, self).__init__()

        a = 32
        b = 7

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, a * b * b),
            nn.Unflatten(1, (a, b, b)),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=2,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=5,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x_pro = self.decoder(z)
        return x_pro

class VoiceEncoder(nn.Module):
    def __init__(
        self,
    ):
        super(VoiceEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(18, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        return x_encoded


class VoiceDecoder(nn.Module):
    def __init__(self, input_dim=20):
        super(VoiceDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 18),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x_pro = self.decoder(z)

        return x_pro

class VannilaEncoder(nn.Module):
    def __init__(
        self,
    ):
        super(VannilaEncoder, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 20),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.linear(x)


class VannilaDecoder(nn.Module):
    def __init__(self, input_dim=20):
        super(VannilaDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(True),
            nn.Linear(20, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 784),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x_pro = self.decoder(z)
        return x_pro


# generated data
class Encoder(nn.Module):
    def __init__(
        self,
    ):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(20, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        return x_encoded


class Decoder(nn.Module):
    def __init__(self, input_dim=20):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
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
    ind = zip(*linear_assignment(w.max() - w))
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, zip(
        *linear_assignment(w.max() - w)
    )


class TripletMarginGeometricJSDLoss(nn.Module):
    def __init__(self, alpha=0.5, margin=1, eps=1e-06):
        super(TripletMarginGeometricJSDLoss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.eps = eps

    @staticmethod
    def _get_alpha_mu_var(mu_p, var_p, mu_q, var_q, a=0.5):
        """Get mean and standard deviation of geometric mean distribution."""
        var_alpha = 1 / ((1 - a) / var_p + a / var_q)
        mu_alpha = var_alpha * ((1 - a) * mu_p / var_p + a * mu_q / var_q)

        return mu_alpha, var_alpha

    def _kl_normal_loss(
        self,
        mu_p: torch.Tensor,
        var_p: torch.Tensor,
        mu_q: torch.Tensor,
        var_q: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the KL divergence between two normal distributions
        with diagonal covariance matrices."""
        r = (
            0.5
            * (
                var_p / var_q + (mu_q - mu_p).pow(2) / var_q - 1 + (var_q / var_p).log()
            ).sum(dim=1)
            + self.eps
        )

        if (r < 0).any():
            print("_kl_normal_loss", r)
        return r

    def gjsd(self, mu_p, log_var_p, mu_q, log_var_q):
        var_p = torch.exp(log_var_p)
        var_q = torch.exp(log_var_q)

        mu_alpha, var_alpha = self._get_alpha_mu_var(
            mu_p, var_p, mu_q, var_q, self.alpha
        )

        kl_1 = self._kl_normal_loss(mu_p, var_p, mu_alpha, var_alpha)
        if (kl_1 < 0).any():
            print("kl_1 gjsd", kl_1)
        kl_2 = self._kl_normal_loss(mu_q, var_q, mu_alpha, var_alpha)
        if (kl_2 < 0).any():
            print("kl_2 gjsd", kl_2)

        res = (1 - self.alpha) * kl_1 + self.alpha * kl_2
        if (res < 0).any():
            print("gjsd nan res", res)

        return res

    def forward(self, mu_a, log_var_a, mu_p, log_var_p, mu_n, log_var_n):
        a = self.gjsd(mu_a, log_var_a, mu_p, log_var_p)
        b = self.gjsd(mu_a, log_var_a, mu_n, log_var_n)
        sqrt_positive_example_distance = torch.sqrt(a + self.eps)
        sqrt_negative_example_distance = torch.sqrt(b + self.eps)

        if (a < 0).any() or sqrt_positive_example_distance.isnan().any():
            print("positive_example_distance", a)

        if (b < 0).any():
            print("sqrt_negative_example_distance", sqrt_negative_example_distance)

        return (
            torch.max(
                torch.zeros_like(
                    sqrt_positive_example_distance,
                    device=sqrt_positive_example_distance.device,
                ),
                sqrt_positive_example_distance
                - sqrt_negative_example_distance
                + self.margin,
            ).mean(),
            sqrt_positive_example_distance.mean(),
            sqrt_negative_example_distance.mean(),
        )


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
            nn.Linear(latent_dim, 10),
            nn.ReLU(True),
            nn.Linear(10, 10),
            nn.ReLU(True),
            nn.Linear(10, number_of_clusters),
        )

    def forward(self, x):
        x = self.scorer(x)
        return x


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

Data_type = Literal["mnist", "generated"]

class Model(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        enc_out_dim=1024,
        latent_dim=10,
        n_clusters=10,
        side_info_dim=10,
        side_info_loss_type="cross_entropy",
        triplet_loss_margin=1.0,
        alpha=1,
        beta=1,
        gamma=1,
        delta=1,
        zeta=1,
        eta=0.5,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.side_info_dim = side_info_dim
        self.side_info_loss_type = side_info_loss_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.zeta = zeta
        self.eta = eta

        self.eps = 1e-9

        self.encoder = encoder()
        self.decoder = decoder(input_dim=latent_dim)

        self.fc_mu_l = nn.Linear(enc_out_dim, latent_dim)
        self.fc_log_sigma2_l = nn.Linear(enc_out_dim, latent_dim)

        self.scorer = Scorer(latent_dim, n_clusters)

        self.side_information = Scorer(latent_dim, side_info_dim)

        self.jsd_triplet_loss = TripletMarginGeometricJSDLoss(
            margin=triplet_loss_margin
        )
    
        def save(self, path="./model.pth", name="model"):	
            torch.save(self.state_dict(), path)	
	

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

    def forward(self, x_a, x_p=None, x_n=None):

        if x_p is not None and x_n is not None:
            encoded_x_a = self.encoder(x_a)
            encoded_x_p = self.encoder(x_p)
            encoded_x_n = self.encoder(x_n)

            mu_x_a, log_sigma2_x_a = self.fc_mu_l(encoded_x_a), self.fc_log_sigma2_l(
                encoded_x_a
            )

            mu_x_p, log_sigma2_x_p = self.fc_mu_l(encoded_x_p), self.fc_log_sigma2_l(
                encoded_x_p
            )
            mu_x_n, log_sigma2_x_n = self.fc_mu_l(encoded_x_n), self.fc_log_sigma2_l(
                encoded_x_n
            )

            z_a = self.sample(mu_x_a, log_sigma2_x_a)
            z_p = self.sample(mu_x_p, log_sigma2_x_p)
            z_n = self.sample(mu_x_n, log_sigma2_x_n)

            # decoded
            reconstructed_x_a = self.decoder(z_a)
            reconstructed_x_p = self.decoder(z_p)
            reconstructed_x_n = self.decoder(z_n)

            side_information_logit_a = self.side_information(z_a)
            side_information_logit_p = self.side_information(z_p)
            side_information_logit_n = self.side_information(z_n)

            logit_a = self.scorer(z_a)
            logit_p = self.scorer(z_p)
            logit_n = self.scorer(z_n)

            return (
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
            )
        else:
            encoded_x_a = self.encoder(x_a)

            mu_x_a, log_sigma2_x_a = self.fc_mu_l(encoded_x_a), self.fc_log_sigma2_l(
                encoded_x_a
            )

            z_a = self.sample(mu_x_a, log_sigma2_x_a)

            logit_a = self.scorer(z_a)

            # decoded
            reconstructed_x_a = self.decoder(z_a)

            side_information_logit_a = self.side_information(z_a)

            return (
                encoded_x_a,
                z_a,
                reconstructed_x_a,
                mu_x_a,
                log_sigma2_x_a,
                logit_a,
                side_information_logit_a,
            )

    @staticmethod
    def kl_divergence(mu, log_var):
        return torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

    def loss_function(
        self,
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
    ):
        reconstruction_loss_x_a = F.binary_cross_entropy(reconstructed_x_a, x_a)
        reconstruction_loss_x_p = F.binary_cross_entropy(reconstructed_x_p, x_p)
        reconstruction_loss_x_n = F.binary_cross_entropy(reconstructed_x_n, x_n)

        reconstruction_loss = self.alpha * (
            reconstruction_loss_x_a + reconstruction_loss_x_p + reconstruction_loss_x_n
        )

        loss_kl_part_a = self.kl_divergence(mu_x_a, log_sigma2_x_a)
        loss_kl_part_p = self.kl_divergence(mu_x_p, log_sigma2_x_p)
        loss_kl_part_n = self.kl_divergence(mu_x_n, log_sigma2_x_n)

        kl_loss = self.beta * (loss_kl_part_a + loss_kl_part_p + loss_kl_part_n)

        (
            triplet_loss,
            positive_example_distance,
            negative_example_distance,
        ) = self.jsd_triplet_loss(
            mu_x_a, log_sigma2_x_a, mu_x_p, log_sigma2_x_p, mu_x_n, log_sigma2_x_n
        )

        triplet_loss_ = self.gamma * triplet_loss

        if self.side_info_loss_type == "binary_cross_entropy":
            side_information_loss_a = F.binary_cross_entropy_with_logits(
                side_information_logit_a.flatten(), side_information_a.float()
            )
            side_information_loss_p = F.binary_cross_entropy_with_logits(
                side_information_logit_p.flatten(), side_information_p.float()
            )
            side_information_loss_n = F.binary_cross_entropy_with_logits(
                side_information_logit_n.flatten(), side_information_n.float()
            )
        elif self.side_info_loss_type == "cross_entropy":
            side_information_loss_a = F.cross_entropy(
                side_information_logit_a, side_information_a
            ).div(math.log(2))
            side_information_loss_p = F.cross_entropy(
                side_information_logit_p, side_information_p
            ).div(math.log(2))
            side_information_loss_n = F.cross_entropy(
                side_information_logit_n, side_information_n
            ).div(math.log(2))
        elif self.side_info_loss_type == "mse":
            side_information_loss_a = F.mse_loss(
                side_information_logit_a.flatten(), side_information_a
            )
            side_information_loss_p = F.mse_loss(
                side_information_logit_p.flatten(), side_information_p
            )
            side_information_loss_n = F.mse_loss(
                side_information_logit_n.flatten(), side_information_n
            )
        else:
            raise ValueError

        side_information_loss = self.delta * (
            side_information_loss_a + side_information_loss_p + side_information_loss_n
        )

        def conditional_entropy(logit):
            if self.n_clusters == 1:
                probs_1 = torch.sigmoid(logit)
                probs_0 = 1 - probs_1

                return -torch.mean(
                    probs_0 * (torch.log(probs_0 + 1e-8)) + probs_1 * (torch.log(probs_1 + 1e-8))
                    )
            else:
                probs = F.softmax(logit, dim=1)
                return -torch.mean(torch.sum(probs * (torch.log(probs + 1e-8)), 1))


        score_given_z_entropy = (
            conditional_entropy(logit_a)
            + conditional_entropy(logit_p)
            + conditional_entropy(logit_n)
        )

        def score_entropy(logit):
            if self.n_clusters == 1:
                probs_1 = torch.sigmoid(logit)
                probs_0 = 1 - probs_1

                score_probs_1 = torch.mean(probs_1, 0)
                score_probs_0 = torch.mean(probs_0, 0)

                return -((score_probs_0 * torch.log(score_probs_0))+((score_probs_1 * torch.log(score_probs_1))))
            else:
                probs = F.softmax(logit, dim=1)
                score_probs = torch.mean(probs, 0)
                return -torch.sum(score_probs * torch.log(score_probs))

        score_entropy = (
            score_entropy(logit_a) + score_entropy(logit_p) + score_entropy(logit_n)
        )

        mutual_info = self.zeta * (
            self.eta * score_entropy - (1 - self.eta) * score_given_z_entropy
        )

        # elbo
        loss = (
            reconstruction_loss
            + kl_loss
            + triplet_loss_
            + side_information_loss
            - mutual_info
        )

        with torch.no_grad():
            positive_metric_distance = (
                (mu_x_a - mu_x_p + self.eps).pow(2).sum(1).sqrt().mean()
            )
            negative_metric_distance = (
                (mu_x_a - mu_x_n + self.eps).pow(2).sum(1).sqrt().mean()
            )

        self.log_dict(
            {
                "loss": loss,
                "recon_loss": reconstruction_loss.mean(),
                "triplet_loss": triplet_loss,
                "kl_loss": kl_loss,
                "positive_example_distance_jsd": positive_example_distance,
                "negative_example_distance_jsd": negative_example_distance,
                "positive_metric_distance": positive_metric_distance,
                "negative_metric_distance": negative_metric_distance,
                "side_information_loss": side_information_loss,
                "mutual_info": mutual_info,
                "score_entropy": score_entropy,
                "score_given_z_entropy": score_given_z_entropy,
            }
        )

        return loss

    def training_step(self, batch, batch_idx):
        par = [p for p in self.parameters() if p.isnan().any()]
        if par:
            print("nan", par)

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
        ) = batch

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
        ) = self(x_a, x_p, x_n)

        elbo = self.loss_function(
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

        return elbo

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        (
            x_a,
            side_information_a,
            y_a,
        ) = batch

        (
            encoded_x_a,
            z_a,
            reconstructed_x_a,
            mu_x_a,
            log_sigma2_x_a,
            logit_a,
            side_information_logit_a,
        ) = self(x_a, None, None)

        if self.n_clusters == 1:
            pred = logit_a
        else: 
            pred = torch.argmax(logit_a, axis=1)

        return pred, logit_a

    def validation_step(self, batch, batch_idx):
        (
            x_a,
            side_information_a,
            y_a,
        ) = batch

        (
            encoded_x_a,
            z_a,
            reconstructed_x_a,
            mu_x_a,
            log_sigma2_x_a,
            logit_a,
            side_information_logit_a,
        ) = self(x_a, None, None)

        pred, logit_a = self.predict_step(batch, batch_idx)

        return (
            x_a,
            mu_x_a,
            log_sigma2_x_a,
            pred,
            y_a,
            side_information_a,
            side_information_logit_a,
            reconstructed_x_a,
            logit_a,
        )

    def validation_epoch_end(self, outputs):
        x_as = []
        mu_s = []
        log_sigma2_s = []
        preds = []
        y_as = []
        side_informations = []
        side_information_logits = []
        reconstructed_xs = []
        logit_as = []

        for out in outputs:
            (
                x_a,
                mu,
                log_sigma2,
                pred,
                y_a,
                side_information,
                side_information_logit,
                reconstructed_x,
                logit_a,
            ) = out
            x_as.append(x_a)
            mu_s.append(mu)
            log_sigma2_s.append(log_sigma2)
            preds.append(pred)
            y_as.append(y_a)
            side_informations.append(side_information)
            side_information_logits.append(side_information_logit)
            reconstructed_xs.append(reconstructed_x)
            logit_as.append(logit_a)

        x_a = torch.cat(x_as).cpu().detach().numpy()
        mu = torch.cat(mu_s)
        log_sigma2_x_a = torch.cat(log_sigma2_s).cpu().detach()
        preds = torch.cat(preds)
        y_a = torch.cat(y_as).cpu().numpy()
        side_information = torch.cat(side_informations)
        side_information_logit = torch.cat(side_information_logits)
        reconstructed_xs = torch.cat(reconstructed_xs).cpu().numpy()
        logit_a = torch.cat(logit_as)
