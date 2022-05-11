import itertools
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import umap
import umap.plot
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import tqdm


# original for mnist
class MnistEncoder(nn.Module):
    def __init__(
        self,
    ):
        super(MnistEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 40),
            nn.ReLU(),
        )

    def forward(self, x):
        if x.isnan().any():
            print("MnistEncoder nan x", x)
        x_encoded = self.encoder(x)
        if x_encoded.isnan().any():
            print("MnistEncoder nan x_encoded", x_encoded)
        return x_encoded


class MnistDecoder(nn.Module):
    def __init__(self, input_dim=20):
        super(MnistDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.ReLU(),
            nn.Linear(40, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Sigmoid(),
        )

    def forward(self, z):
        if z.isnan().any():
            print("MnistDecoder nan z", z)
        x_pro = self.decoder(z)
        if x_pro.isnan().any():
            print("MnistDecoder nan x_pro", x_pro)
        return x_pro


# mnist style for generated data
# class Encoder(nn.Module):
#     def __init__(
#         self,
#     ):
#         super(Encoder, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Linear(200, 500),
#             nn.ReLU(),
#             nn.Linear(500, 500),
#             nn.ReLU(),
#             nn.Linear(500, 2000),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         x_encoded = self.encoder(x)
#         return x_encoded


# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()

#         self.decoder = nn.Sequential(
#             nn.Linear(10, 2000),
#             nn.ReLU(),
#             nn.Linear(2000, 500),
#             nn.ReLU(),
#             nn.Linear(500, 500),
#             nn.ReLU(),
#             nn.Linear(500, 200),
#         )

#     def forward(self, z):
#         x_pro = self.decoder(z)

#         return x_pro


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


class GMM_Encoder(nn.Module):
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


class GMM_Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x_pro = self.decoder(z)

        return x_pro


class Pi(nn.Module):
    def __init__(self, n_clusters):
        super(Pi, self).__init__()

        self.pi_ = nn.Parameter(
            torch.FloatTensor(
                n_clusters,
            ).fill_(1)
            / n_clusters,
            requires_grad=True,
        )

    def forward(self):
        return F.softmax(self.pi_, dim=0)


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = int(max(Y_pred.max(), Y.max()) + 1)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], int(Y[i])] += 1
    ind = zip(*linear_assignment(w.max() - w))
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size


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
        r = 0.5 * (
            var_p / var_q + (mu_q - mu_p).pow(2) / var_q - 1 + (var_q / var_p).log()
        ).sum(dim=1)

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

        self.fc1 = nn.Linear(latent_dim, number_of_clusters)

    def forward(self, x):
        x = self.fc1(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape

        # Calculate distances
        distances = (
            torch.sum(inputs**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(inputs, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

Kl_loss_type = Literal["vade", "vib-gmm", "vae", "vqvae"]
Mu_c_init = Literal["zero", "normal"]
Log_sigma2_c_init = Literal["zero", "normal"]
Optimizers_type = Literal["adam", "adam_scheduler"]
Data_type = Literal["mnist", "generated"]
Prediction_type = ["gmm", "vqvae", "mutual_info"]


class Model(pl.LightningModule):
    def __init__(
        self,
        enc_out_dim=2000,
        latent_dim=10,
        n_clusters=10,
        triplet_loss_margin=1.0,
        n_samples=1000,
        alpha=1,
        beta=1,
        gamma=1,
        delta=1,
        zeta=1,
        triplet_loss_type="TripletMarginGeometricJSDLoss",
        kl_loss_type: Kl_loss_type = "vade",
        mu_c_init: Mu_c_init = "zero",
        log_sigma2_c_init: Log_sigma2_c_init = "zero",
        optimizers_type: Optimizers_type = "adam",
        side_information_enabled=True,
        prediction_type: Prediction_type = "mutual_info",
        data_type: Data_type = "mnist",
    ):
        super().__init__()

        self.save_hyperparameters()
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.zeta = zeta
        self.triplet_loss_type = triplet_loss_type
        self.kl_loss_type = kl_loss_type
        self.mu_c_init = mu_c_init
        self.log_sigma2_c_init = log_sigma2_c_init
        self.optimizers_type = optimizers_type
        self.side_information_enabled = side_information_enabled
        self.prediction_type = prediction_type
        self.data_type = data_type

        if self.mu_c_init not in ["zero", "normal"]:
            raise ValueError('mu_c_init must be in ["zero", "normal"]')

        if self.log_sigma2_c_init not in ["zero", "normal"]:
            raise ValueError('log_sigma2_c_init must be in ["zero", "normal"]')

        if self.optimizers_type not in ["adam", "adam_scheduler"]:
            raise ValueError('optimizers_type must be in ["adam", "adam_scheduler"]')

        self.eps = 1e-9

        assert type(self.beta) in [int, float, list]

        # encoder, decoder
        if self.data_type == "generated":
            self.encoder = Encoder()
            self.decoder = Decoder(input_dim=latent_dim)
        elif self.data_type == "mnist":
            self.encoder = MnistEncoder()
            self.decoder = MnistDecoder(input_dim=latent_dim)

        self._vq_vae = VectorQuantizer(
            num_embeddings=n_clusters, embedding_dim=latent_dim, commitment_cost=0.25
        )

        # distribution parameters
        self.fc_mu_l = nn.Linear(enc_out_dim, latent_dim)
        self.fc_log_sigma2_l = nn.Linear(enc_out_dim, latent_dim)

        self.scorer = Scorer(latent_dim, n_clusters)
        self.side_information = Scorer(latent_dim, n_clusters)

        self.pi = Pi(n_clusters)

        if self.mu_c_init == "zero":
            self.mu_c = nn.Parameter(
                torch.FloatTensor(n_clusters, latent_dim).fill_(0), requires_grad=True
            )
        elif self.mu_c_init == "normal":
            mu_c = torch.empty(n_clusters, latent_dim)
            torch.nn.init.normal_(mu_c)
            self.mu_c = nn.Parameter(mu_c, requires_grad=True)

        if self.log_sigma2_c_init == "zero":
            self.log_sigma2_c = nn.Parameter(
                torch.FloatTensor(n_clusters, latent_dim).fill_(0), requires_grad=True
            )
        elif self.log_sigma2_c_init == "normal":
            log_sigma2_c = torch.empty(n_clusters, latent_dim)
            torch.nn.init.normal_(log_sigma2_c)
            self.log_sigma2_c = nn.Parameter(
                0.25 * torch.abs(log_sigma2_c), requires_grad=True
            )

        self.jsd_triplet_loss = TripletMarginGeometricJSDLoss(
            margin=triplet_loss_margin
        )
        self.metric_triplet_loss = nn.TripletMarginLoss(margin=triplet_loss_margin)

    def on_train_start(self):
        self.hparams.update(self.trainer.datamodule.get_summery())
        self.logger.log_hyperparams(self.hparams)

    def configure_optimizers(self):
        if self.optimizers_type == "adam":
            return torch.optim.Adam(self.parameters(), lr=2e-3)
        elif self.optimizers_type == "adam_scheduler":
            optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.9
            )
            return [optimizer], [lr_scheduler]

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):
        gs = []
        for c in range(self.n_clusters):
            gs.append(
                self.gaussian_pdf_log(
                    x, mus[c : c + 1, :], log_sigma2s[c : c + 1, :]
                ).view(-1, 1)
            )
        return torch.cat(gs, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):
        return -0.5 * (
            torch.sum(
                np.log(np.pi * 2)
                + log_sigma2
                + (x - mu).pow(2) / torch.exp(log_sigma2),
                1,
            )
        )

    def pre_train(self, dataloader, pre_epoch=10):

        if not os.path.exists("./pretrain_model.pk"):

            Loss = nn.MSELoss()
            opti = Adam(self.parameters())

            print("Pretraining......")
            epoch_bar = tqdm(range(pre_epoch))
            for _ in epoch_bar:
                L = 0
                for batch in dataloader.train_dataloader():

                    (
                        x_a,
                        x_p,
                        x_n,
                        _,
                        _,
                        _,
                        y_a,
                        y_p,
                        y_n,
                    ) = batch

                    x_encoded = self.encoder(x_a)
                    mu_x, log_sigma2_x = self.fc_mu_l(x_encoded), self.fc_log_sigma2_l(
                        x_encoded
                    )

                    x_ = self.decoder(mu_x)
                    loss = Loss(x_a, x_)

                    L += loss.detach().cpu().numpy()

                    opti.zero_grad()
                    loss.backward()
                    opti.step()

                epoch_bar.write(
                    "L2={:.4f}".format(L / len(dataloader.train_dataloader()))
                )

            self.fc_log_sigma2_l.load_state_dict(self.fc_mu_l.state_dict())

            Z_ = []
            Y_ = []
            log_sigma2_xs_ = []
            with torch.no_grad():
                for batch in dataloader.train_dataloader():
                    (
                        x_a,
                        x_p,
                        x_n,
                        _,
                        _,
                        _,
                        y_a,
                        y_p,
                        y_n,
                    ) = batch

                    x_encoded = self.encoder(x_a)
                    mu_x, log_sigma2_x = self.fc_mu_l(x_encoded), self.fc_log_sigma2_l(
                        x_encoded
                    )
                    assert F.mse_loss(mu_x, log_sigma2_x) == 0
                    Z_.append(mu_x)
                    Y_.append(y_a)

            Z = torch.cat(Z_, 0).detach().cpu().numpy()
            Y = torch.cat(Y_, 0).detach().numpy()

            gmm = GaussianMixture(n_components=self.n_clusters, covariance_type="diag")

            pre = gmm.fit_predict(Z)
            print("Acc={:.4f}%".format(cluster_acc(pre, Y) * 100))

            self.pi.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
            self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
            self.log_sigma2_c.data = torch.log(
                torch.from_numpy(gmm.covariances_).cuda().float()
            )

            torch.save(self.state_dict(), "./pretrain_model.pk")

            return Z_, Y_, Z, Y, gmm, pre

        else:

            self.load_state_dict(torch.load("./pretrain_model.pk"))

    def sample_from_all_gaussians(self, n=16):
        zs = []
        for mu, log_var in zip(self.mu_c, self.log_sigma2_c):
            std = torch.exp(log_var / 2)
            p = torch.distributions.Normal(mu, std)
            zs.append(p.rsample((n,)))
        return torch.cat(zs)

    def sample_from_mixture(self, n_total=16):
        pi = self.pi().cpu().detach().numpy()
        cs = np.random.choice(pi.size, n_total, p=pi)
        c, counts = np.unique(cs, return_counts=True)

        zs = []
        for c, count in zip(c, counts):
            mu = self.mu_c[c]
            log_var = self.log_sigma2_c[c]
            std = torch.exp(log_var / 2)
            p = torch.distributions.Normal(mu, std)
            zs.append(p.rsample((count,)))
        return torch.cat(zs), sorted(cs)

    @staticmethod
    def sample(mu, log_sigma2):
        std = torch.exp(log_sigma2 / 2)
        z = torch.randn_like(mu) * std + mu

        return z

    def forward(self, x_a, x_p, x_n):
        encoded_x_a = self.encoder(x_a)
        encoded_x_p = self.encoder(x_p)
        encoded_x_n = self.encoder(x_n)

        if encoded_x_a.isnan().any():
            print("encoded_x_a", encoded_x_a)

        # encode x to get the mu and variance parameters
        mu_x_a, log_sigma2_x_a = self.fc_mu_l(encoded_x_a), self.fc_log_sigma2_l(
            encoded_x_a
        )

        if mu_x_a.isnan().any():
            print("mu_x_a", mu_x_a)

        if log_sigma2_x_a.isnan().any():
            print("log_sigma2_x_a", log_sigma2_x_a)

        mu_x_p, log_sigma2_x_p = self.fc_mu_l(encoded_x_p), self.fc_log_sigma2_l(
            encoded_x_p
        )
        mu_x_n, log_sigma2_x_n = self.fc_mu_l(encoded_x_n), self.fc_log_sigma2_l(
            encoded_x_n
        )

        if not self.kl_loss_type == "vqvae":
            # sample z from q
            z_a = self.sample(mu_x_a, log_sigma2_x_a)
            z_p = self.sample(mu_x_p, log_sigma2_x_p)
            z_n = self.sample(mu_x_n, log_sigma2_x_n)

            vq_vae_loss = 0
            perplexity = 0
        else:
            vq_vae_loss_a, z_a, perplexity_a, _ = self._vq_vae(mu_x_a)
            vq_vae_loss_p, z_p, perplexity_p, _ = self._vq_vae(mu_x_p)
            vq_vae_loss_n, z_n, perplexity_n, _ = self._vq_vae(mu_x_n)

            vq_vae_loss = vq_vae_loss_a + vq_vae_loss_p + vq_vae_loss_n
            perplexity = perplexity_a + perplexity_p + perplexity_n

        # decoded
        reconstructed_x_a = self.decoder(z_a)
        reconstructed_x_p = self.decoder(z_p)
        reconstructed_x_n = self.decoder(z_n)

        if reconstructed_x_a.isnan().any():
            print("reconstructed_x_a", reconstructed_x_a)

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
            vq_vae_loss,
            perplexity,
        )

    @staticmethod
    def vib_gmm_kl(mu_enc_, log_sigma_enc_, gmm_coef_, gmm_mu_, gmm_log_sigma_, eps):

        sigma_enc_ = torch.exp(log_sigma_enc_)
        gmm_sigma_ = torch.exp(gmm_log_sigma_)

        GMM_COEF = gmm_coef_
        GMM_MU = gmm_mu_
        GMM_SIGMA = gmm_sigma_

        MU_ENC = mu_enc_
        SIGMA_ENC = sigma_enc_

        KL = 0.5 * torch.sum(
            torch.square((MU_ENC.unsqueeze(1) - GMM_MU.unsqueeze(0)) / GMM_SIGMA)
            + torch.log(GMM_SIGMA.unsqueeze(0) / SIGMA_ENC.unsqueeze(1))
            - 1
            + SIGMA_ENC.unsqueeze(1) / GMM_SIGMA.unsqueeze(0),
            dim=2,
        )

        KL_variational = -torch.logsumexp(torch.log(GMM_COEF) - KL, axis=1)

        return torch.mean(KL_variational)

    def vade_kl(self, z, mu_x, log_sigma2_x):
        p_c_z = (
            torch.exp(
                torch.log(self.pi().unsqueeze(0))
                + self.gaussian_pdfs_log(z, self.mu_c, self.log_sigma2_c)
            )
            + self.eps
        )

        gamma_c = p_c_z / (p_c_z.sum(1).view(-1, 1))  # batch_size*Clusters

        second_term = 0.5 * torch.mean(
            torch.sum(
                gamma_c
                * torch.sum(
                    self.log_sigma2_c.unsqueeze(0)
                    + torch.exp(
                        log_sigma2_x.unsqueeze(1) - self.log_sigma2_c.unsqueeze(0)
                    )
                    + (mu_x.unsqueeze(1) - self.mu_c.unsqueeze(0)).pow(2)
                    / torch.exp(self.log_sigma2_c.unsqueeze(0)),
                    2,
                ),
                1,
            )
        )

        third_term = torch.mean(
            torch.sum(gamma_c * torch.log(self.pi().unsqueeze(0) / (gamma_c)), 1)
        )

        fourth_term = 0.5 * torch.mean(torch.sum(1 + log_sigma2_x, 1))

        self.log_dict(
            {
                "second_term": second_term.mean(),
                "third_term": -third_term.mean(),
                "fourth_term": -fourth_term.mean(),
            }
        )

        return second_term - (third_term + fourth_term)

    @staticmethod
    def vae_kl(mu, log_var):
        return torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

    def vib_loss(self, logit, y, mu, std, w):
        class_loss = F.cross_entropy(logit, y).div(math.log(2))
        info_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(
            1
        ).mean().div(math.log(2))
        total_loss = class_loss + w * info_loss

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
        vq_vae_loss,
    ):

        d = x_a.size(1)

        if reconstructed_x_a.isnan().any():
            print(reconstructed_x_a)

        reconstruction_loss_x_a = F.binary_cross_entropy(reconstructed_x_a, x_a) * d
        reconstruction_loss_x_p = F.binary_cross_entropy(reconstructed_x_p, x_p) * d
        reconstruction_loss_x_n = F.binary_cross_entropy(reconstructed_x_n, x_n) * d

        if self.kl_loss_type == "vade":
            loss_kl_part_a = self.vade_kl(z_a, mu_x_a, log_sigma2_x_a)
            loss_kl_part_p = self.vade_kl(z_p, mu_x_p, log_sigma2_x_p)
            loss_kl_part_n = self.vade_kl(z_n, mu_x_n, log_sigma2_x_n)
        elif self.kl_loss_type == "vib-gmm":
            loss_kl_part_a = self.vib_gmm_kl(
                mu_x_a,
                log_sigma2_x_a,
                self.pi(),
                self.mu_c,
                self.log_sigma2_c,
                self.eps,
            )
            loss_kl_part_p = self.vib_gmm_kl(
                mu_x_p,
                log_sigma2_x_p,
                self.pi(),
                self.mu_c,
                self.log_sigma2_c,
                self.eps,
            )
            loss_kl_part_n = self.vib_gmm_kl(
                mu_x_n,
                log_sigma2_x_n,
                self.pi(),
                self.mu_c,
                self.log_sigma2_c,
                self.eps,
            )
        elif self.kl_loss_type == "vae":
            loss_kl_part_a = self.vae_kl(mu_x_a, log_sigma2_x_a)
            loss_kl_part_p = self.vae_kl(mu_x_p, log_sigma2_x_p)
            loss_kl_part_n = self.vae_kl(mu_x_n, log_sigma2_x_n)
        elif self.kl_loss_type == "vqvae":
            pass
        else:
            raise ValueError

        reconstruction_loss = self.alpha * (
            reconstruction_loss_x_a + reconstruction_loss_x_p + reconstruction_loss_x_n
        )

        beta = (
            self.beta[self.current_epoch] if isinstance(self.beta, list) else self.beta
        )

        if self.kl_loss_type == "vqvae":
            kl_loss = beta * vq_vae_loss
        else:
            kl_loss = beta * (loss_kl_part_a + loss_kl_part_p + loss_kl_part_n)

        if self.triplet_loss_type == "TripletMarginGeometricJSDLoss":
            (
                triplet_loss,
                positive_example_distance,
                negative_example_distance,
            ) = self.jsd_triplet_loss(
                mu_x_a, log_sigma2_x_a, mu_x_p, log_sigma2_x_p, mu_x_n, log_sigma2_x_n
            )

            with torch.no_grad():
                metric_triplet_loss = self.metric_triplet_loss(mu_x_a, mu_x_p, mu_x_n)

            triplet_loss_ = self.gamma * triplet_loss

        elif self.triplet_loss_type == "TripletMarginLoss":
            metric_triplet_loss = self.metric_triplet_loss(mu_x_a, mu_x_p, mu_x_n)

            with torch.no_grad():
                (
                    triplet_loss,
                    positive_example_distance,
                    negative_example_distance,
                ) = self.jsd_triplet_loss(
                    mu_x_a,
                    log_sigma2_x_a,
                    mu_x_p,
                    log_sigma2_x_p,
                    mu_x_n,
                    log_sigma2_x_n,
                )

            triplet_loss_ = self.gamma * metric_triplet_loss

        side_information_loss = 0
        if self.side_information_enabled:
            side_information_loss_a = F.cross_entropy(
                side_information_logit_a, side_information_a
            ).div(math.log(2))
            side_information_loss_p = F.cross_entropy(
                side_information_logit_p, side_information_p
            ).div(math.log(2))
            side_information_loss_n = F.cross_entropy(
                side_information_logit_n, side_information_n
            ).div(math.log(2))

            side_information_loss = self.delta * (
                side_information_loss_a
                + side_information_loss_p
                + side_information_loss_n
            )

        def conditinal_entropy(logit):
            probs = F.softmax(logit, dim=1)
            return -torch.mean(torch.sum(probs * (torch.log(probs + 1e-5)), 1))

        score_given_z_entropy = (
            conditinal_entropy(logit_a)
            + conditinal_entropy(logit_p)
            + conditinal_entropy(logit_n)
        )

        def score_entropy(logit):
            probs = F.softmax(logit, dim=1)
            score_probs = torch.mean(probs, 0)
            return -torch.sum(score_probs * torch.log(score_probs))

        score_entropy = (
            score_entropy(logit_a) + score_entropy(logit_p) + score_entropy(logit_n)
        )

        lamda = 0.1

        mutual_info = self.zeta * (score_entropy - score_given_z_entropy)
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
                "beta": beta,
                "metric_triplet_loss": metric_triplet_loss,
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
            vq_vae_loss,
            perplexity,
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
            vq_vae_loss,
        )

        return elbo

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
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
            vq_vae_loss,
            perplexity,
        ) = self(x_a, x_p, x_n)

        if self.prediction_type == "vqvae":
            pred = (
                (
                    torch.sum(z_a**2, dim=1, keepdim=True)
                    + torch.sum(self._vq_vae._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(z_a, self._vq_vae._embedding.weight.t())
                )
                .argmin(axis=1)
                .detach()
                .cpu()
                .numpy()
            )
        elif self.prediction_type == "gmm":
            p_c_z = (
                (
                    torch.exp(
                        torch.log(self.pi().unsqueeze(0))
                        + self.gaussian_pdfs_log(z_a, self.mu_c, self.log_sigma2_c)
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )
            pred = np.argmax(p_c_z, axis=1)
        elif self.prediction_type == "mutual_info":
            pred = torch.argmax(logit_a, axis=1).detach().cpu().numpy()
        else:
            raise ValueError()

        return (
            pred,
            torch.argmax(logit_a, axis=1).detach().cpu().numpy(),
        )

    def validation_step(self, batch, batch_idx):
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
            vq_vae_loss,
            perplexity,
        ) = self(x_a, x_p, x_n)

        pred, score = self.predict_step(batch, batch_idx)

        return mu_x_a, log_sigma2_x_a, pred, score, y_a

    def validation_epoch_end(self, outputs):

        mu_x_as = []
        log_sigma2_x_as = []
        preds = []
        scores = []
        y_as = []

        for out in outputs:
            mu_x_a, log_sigma2_x_a, pred, score, y_a = out
            mu_x_as.append(mu_x_a)
            log_sigma2_x_as.append(log_sigma2_x_a)
            preds.append(pred)
            scores.append(score)
            y_as.append(y_a)

        mu_x_a = torch.cat(mu_x_as).cpu().detach().numpy()
        log_sigma2_x_a = torch.cat(log_sigma2_x_as).cpu().detach()
        preds = np.concatenate(preds, 0)
        score = np.concatenate(scores, 0)
        y_a = torch.cat(y_as).cpu().numpy()

        self.logger.experiment.log(
            {
                "preds": preds,
                "mu_x_a": mu_x_a,
                "log_sigma2_x_a": log_sigma2_x_a.numpy(),
                "log_sigma2_x_a norm": torch.linalg.norm(log_sigma2_x_a, axis=1).mean(),
                "y_a": y_a,
                "score": score,
            }
        )

        mapper = umap.UMAP().fit(mu_x_a)
        mapper_vq_vae = umap.UMAP().fit(
            self._vq_vae._embedding.weight.data.cpu().detach()
        )

        self.logger.experiment.log(
            {
                "acc": cluster_acc(preds, y_a),
                "U-map q(z|x)": wandb.Image(
                    umap.plot.points(mapper, labels=np.squeeze(y_a))
                ),
                "U-map z": wandb.Image(self.umap_embedding()),
                "U-map q(z|x) score": wandb.Image(
                    umap.plot.points(mapper, labels=np.squeeze(preds))
                ),
                "vq_vae embeddings": wandb.Image(umap.plot.points(mapper_vq_vae)),
            }
        )

    def umap_embedding(self, n=10000):
        z, cs = self.sample_from_mixture(
            n_total=n,
        )

        mapper = umap.UMAP().fit(z.cpu().detach().numpy())
        return umap.plot.points(mapper, labels=np.array(cs))

    def plot_instances(self):
        from matplotlib.pyplot import figure, imshow
        from torchvision.utils import make_grid

        figure(figsize=(5, 10), dpi=300)

        # SAMPLE IMAGES
        z = self.sample_from_all_gaussians(n=8)
        with torch.no_grad():
            pred = self.decoder(z.to(vade.device)).cpu()

        unflatten = nn.Unflatten(1, (28, 28))

        img = make_grid(unflatten(pred)[:, None, :])

        # PLOT IMAGES
        # imshow(img.permute(1, 2, 0))
        return img

    def training_epoch_end(self, outputs):

        self.logger.experiment.log(
            {
                "pi": wandb.Histogram(self.pi().cpu().detach().numpy()),
                "mu_c": wandb.Histogram(self.mu_c.cpu().detach().numpy()),
                "log_sigma2_c": wandb.Histogram(
                    self.log_sigma2_c.cpu().detach().numpy()
                ),
            },
        )
