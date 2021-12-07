import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
from torch.optim import Adam
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import umap
import umap.plot
from scipy.optimize import linear_sum_assignment as linear_assignment
import wandb


# class Encoder(nn.Module):
#     def __init__(
#         self,
#     ):
#         super(Encoder, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Linear(784, 500),
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
#             nn.Linear(500, 784),
#             nn.Sigmoid(),
#         )

#     def forward(self, z):
#         x_pro = self.decoder(z)

#         return x_pro


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
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.Sigmoid()
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
    def __init__(self, alpha=0.5, margin=1.):
        super(TripletMarginGeometricJSDLoss, self).__init__()
        self.alpha = alpha
        self.margin = margin

    @staticmethod
    def _get_alpha_mu_var(mu_p, var_p, mu_q, var_q, a=0.5):
        """Get mean and standard deviation of geometric mean distribution."""
        var_alpha = 1 / ((1 - a) / var_p + a / var_q)
        mu_alpha = var_alpha * ((1 - a) * mu_p / var_p + a * mu_q / var_q)

        return mu_alpha, var_alpha

    def _kl_normal_loss(
            self,
            mu_p: torch.Tensor,
            log_var_p: torch.Tensor,
            mu_q: torch.Tensor,
            log_var_q: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the KL divergence between two normal distributions
        with diagonal covariance matrices."""
        r = 0.5 * (
                log_var_p.exp() / log_var_q.exp()
                + (mu_q - mu_p).pow(2) / log_var_q.exp()
                - 1
                + (log_var_q - log_var_p)
        ).sum(dim=1)

        return r

    def gjsd(self, mu_p, log_var_p, mu_q, log_var_q):
        var_p = torch.exp(log_var_p)
        var_q = torch.exp(log_var_q)

        mu_alpha, var_alpha = self._get_alpha_mu_var(
            mu_p, var_p, mu_q, var_q, self.alpha
        )
        log_var_alpha = torch.log(var_alpha)

        kl_1 = self._kl_normal_loss(mu_p, log_var_p, mu_alpha, log_var_alpha)
        kl_2 = self._kl_normal_loss(mu_q, log_var_q, mu_alpha, log_var_alpha)

        return ((1 - self.alpha) * kl_1 + self.alpha * kl_2)

    def forward(self, mu_a, log_var_a, mu_p, log_var_p, mu_n, log_var_n):
        sqrt_positive_example_distance = torch.sqrt(self.gjsd(mu_a, log_var_a, mu_p, log_var_p))
        sqrt_negative_example_distance = torch.sqrt(self.gjsd(mu_a, log_var_a, mu_n, log_var_n))

        return (
            torch.max(
                torch.zeros_like(sqrt_positive_example_distance, device=sqrt_positive_example_distance.device),
                sqrt_positive_example_distance - sqrt_negative_example_distance + self.margin,
                ).mean(),
            sqrt_positive_example_distance.mean(),
            sqrt_negative_example_distance.mean(),
        )


class VaDE(pl.LightningModule):
    def __init__(
            self,
            enc_out_dim=2000,
            latent_dim=10,
            n_clusters=10,
            triplet_loss_margin=1.0,
            alpha=1,
            beta=1,
            gamma=1
    ):
        super().__init__()

        self.save_hyperparameters()
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = 1e-10

        # encoder, decoder
        self.encoder = Encoder()
        self.decoder = Decoder()

        # distribution parameters
        self.fc_mu_l = nn.Linear(enc_out_dim, latent_dim)
        self.fc_log_sigma2_l = nn.Linear(enc_out_dim, latent_dim)

        self.pi = Pi(n_clusters)

        self.mu_c = nn.Parameter(
            torch.FloatTensor(n_clusters, latent_dim).fill_(0), requires_grad=True
        )
        self.log_sigma2_c = nn.Parameter(
            torch.FloatTensor(n_clusters, latent_dim).fill_(0), requires_grad=True
        )

        self.triplet_loss = TripletMarginGeometricJSDLoss(margin=triplet_loss_margin)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-3)

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
            with torch.no_grad():
                for batch in dataloader.train_dataloader():
                    (
                        x_a,
                        x_p,
                        x_n,
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


        # sample z from q
        z_a = self.sample(mu_x_a, log_sigma2_x_a)
        z_p = self.sample(mu_x_p, log_sigma2_x_p)
        z_n = self.sample(mu_x_n, log_sigma2_x_n)

        # decoded
        reconstructed_x_a = self.decoder(z_a)
        if reconstructed_x_a.isnan().any():
            print("reconstructed_x_a", reconstructed_x_a)
        reconstructed_x_p = self.decoder(z_p)
        reconstructed_x_n = self.decoder(z_n)

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
        )

    def loss_kl_part(self, z, mu_x, log_sigma2_x):
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
    ):

        d = x_a.size(1)

        if reconstructed_x_a.isnan().any():
            print(reconstructed_x_a)

        reconstruction_loss_x_a = F.binary_cross_entropy(reconstructed_x_a, x_a) * d
        reconstruction_loss_x_p = F.binary_cross_entropy(reconstructed_x_p, x_p) * d
        reconstruction_loss_x_n = F.binary_cross_entropy(reconstructed_x_n, x_n) * d

        loss_kl_part_a = self.loss_kl_part(z_a, mu_x_a, log_sigma2_x_a)
        loss_kl_part_p = self.loss_kl_part(z_p, mu_x_p, log_sigma2_x_p)
        loss_kl_part_n = self.loss_kl_part(z_n, mu_x_n, log_sigma2_x_n)

        triplet_loss, positive_example_distance, negative_example_distance = self.triplet_loss(
            mu_x_a, mu_x_p, mu_x_n, log_sigma2_x_a, log_sigma2_x_p, log_sigma2_x_n
        )

        reconstruction_loss = self.alpha * (
                reconstruction_loss_x_a
                + reconstruction_loss_x_p
                + reconstruction_loss_x_n
        )

        kl_loss = self.beta * (
                + loss_kl_part_a
                + loss_kl_part_p
                + loss_kl_part_n
        )

        triplet_loss_ = self.gamma * triplet_loss


        # elbo
        loss = reconstruction_loss + kl_loss + triplet_loss_

        self.log_dict(
            {
                "loss": loss,
                "recon_loss": reconstruction_loss.mean(),
                "triplet_loss": triplet_loss,
                "positive_example_distance": positive_example_distance,
                "negative_example_distance": negative_example_distance
            }
        )

        return loss

    def training_step(self, batch, batch_idx):
        (
            x_a,
            x_p,
            x_n,
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
        )

        return elbo

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        (
            x_a,
            x_p,
            x_n,
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
        ) = self(x_a, x_p, x_n)

        p_c_z = (
            torch.exp(
                torch.log(self.pi().unsqueeze(0))
                + self.gaussian_pdfs_log(z_a, self.mu_c, self.log_sigma2_c)
            )
        ).detach().cpu().numpy()

        return np.argmax(p_c_z,axis=1)

    def validation_step(self, batch, batch_idx):
        (
            x_a,
            x_p,
            x_n,
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
        ) = self(x_a, x_p, x_n)

        return mu_x_a, self.predict_step(batch, batch_idx), y_a

    def validation_epoch_end(self, outputs):

        mu_x_as = []
        preds = []
        y_as = []

        for out in outputs:
            mu_x_a, pred, y_a = out

            mu_x_as.append(mu_x_a)
            preds.append(pred)
            y_as.append(y_a)

        mu_x_a = torch.cat(mu_x_as).cpu().detach().numpy()
        pred = np.concatenate(preds,0)
        y_a = torch.cat(y_as).cpu().numpy()

        mapper = umap.UMAP().fit(mu_x_a)

        self.logger.experiment.log(
            {
                "pred": pred,
                "mu_x_a": mu_x_a,
                "y_a":y_a
            }
        )

        self.logger.experiment.log(
            {
                "acc": cluster_acc(pred, y_a),
                "U-map q(z|x)": wandb.Image(
                    umap.plot.points(mapper, labels=np.squeeze(y_a))
                ),
                "U-map z": wandb.Image(self.umap_embedding()),
                # "Image": wandb.Image(self.plot_instances()),
            }
        )


    def umap_embedding(self, n=10000):
        z, cs = self.sample_from_mixture(
            n_total=n,
        )

        mapper = umap.UMAP().fit(z.cpu().detach().numpy())
        return umap.plot.points(mapper, labels=np.array(cs))

    def plot_instances(self):
        from matplotlib.pyplot import imshow, figure
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
