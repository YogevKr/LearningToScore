from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
import umap.plot
from clusters_matching import match
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import functional as FM
from torch.autograd import Function
from torch.nn.modules.loss import _Loss

import wandb


class MutualInformationLoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(MutualInformationLoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        assert not target.requires_grad

        return F.mse_loss(
            input, target, size_average=self.size_average, reduce=self.reduce
        )


class MultiTaskLoss(nn.Module):
    """https://arxiv.org/abs/1705.07115"""

    def __init__(self, is_regression, reduction="none"):
        super(MultiTaskLoss, self).__init__()
        self.is_regression = is_regression
        self.n_tasks = len(is_regression)
        self.log_vars = nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction

    def forward(self, losses):
        dtype = losses.dtype
        device = losses.device
        stds = (torch.exp(self.log_vars) ** (1 / 2)).to(device).to(dtype)
        self.is_regression = self.is_regression.to(device).to(dtype)
        coeffs = 1 / ((self.is_regression + 1) * (stds ** 2))
        multi_task_losses = coeffs * losses + torch.log(stds)

        if self.reduction == "sum":
            multi_task_losses = multi_task_losses.sum()
        if self.reduction == "mean":
            multi_task_losses = multi_task_losses.mean()

        return multi_task_losses


class DiffArgMax(Function):
    @staticmethod
    def forward(ctx, input):
        indices = torch.argmax(input, dim=1)
        one_hot = F.one_hot(indices, num_classes=input.shape[1]).float()
        ctx.save_for_backward(one_hot)
        return one_hot * input

    @staticmethod
    def backward(ctx, grad_output):
        (one_hot,) = ctx.saved_tensors
        return grad_output * one_hot


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dims
        current_dim = input_dim
        self.layers: Union[List, nn.ModuleList] = nn.ModuleList()
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            current_dim = h_dim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(Encoder, self).__init__()
        self.encoder = Net(input_dim, output_dim, hidden_dims)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(Decoder, self).__init__()
        self.decoder = Net(input_dim, output_dim, hidden_dims)

    def forward(self, x):
        return self.decoder(x)


class Scorer(nn.Module):
    def __init__(
        self, latent_dim, number_of_clusters, last_activation_function="softmax"
    ):
        super(Scorer, self).__init__()
        self.latent_dim = latent_dim
        self.number_of_clusters = number_of_clusters
        self.last_activation_function = last_activation_function

        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, number_of_clusters)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.last_activation_function == "softmax":
            x = F.softmax(self.fc2(x), dim=1)
        elif self.last_activation_function == "argmax":
            x = DiffArgMax.apply(self.fc2(x))

        return x


class ScorerToSideInformation(nn.Module):
    def __init__(self, number_of_clusters):
        super(ScorerToSideInformation, self).__init__()
        self.number_of_clusters = number_of_clusters
        self.fc1 = nn.Linear(number_of_clusters, number_of_clusters)

    def forward(self, x):
        x = self.fc1(x)
        return x


class Model(LightningModule):
    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_hidden_dims,
        triplet_loss_margin=1.0,
        number_of_clusters=10,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        scorer_last_activation_function="softmax",
        triplet_loss_objective="latent_space",
        uncertainty_loss=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.reconstruction_loss = nn.MSELoss()
        self.triplet_loss_margin = triplet_loss_margin
        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_loss_margin, p=2)

        self.side_information_loss = nn.CrossEntropyLoss()

        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = self.encoder_hidden_dims[::-1]

        self.encoder = Encoder(input_dim, latent_dim, self.encoder_hidden_dims)
        self.decoder = Decoder(latent_dim, input_dim, self.decoder_hidden_dims)

        self.scorer = Scorer(
            latent_dim=latent_dim,
            number_of_clusters=number_of_clusters,
            last_activation_function=scorer_last_activation_function,
        )
        self.score_to_side_information = ScorerToSideInformation(
            number_of_clusters=number_of_clusters
        )

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.number_of_clusters = number_of_clusters

        self.triplet_loss_objective = triplet_loss_objective
        self.uncertainty_loss = uncertainty_loss

        if self.uncertainty_loss:
            self.is_regression = torch.Tensor([True, True, False])
            self.multitaskloss_instance = MultiTaskLoss(
                self.is_regression, reduction="mean"
            )

    def forward(self, x_a, x_p, x_n):
        encoded_x_a = self.encoder(x_a)
        encoded_x_p = self.encoder(x_p)
        encoded_x_n = self.encoder(x_n)

        reconstructed_x_a = self.decoder(encoded_x_a)
        reconstructed_x_p = self.decoder(encoded_x_p)
        reconstructed_x_n = self.decoder(encoded_x_n)

        return (
            encoded_x_a,
            encoded_x_p,
            encoded_x_n,
            reconstructed_x_a,
            reconstructed_x_p,
            reconstructed_x_n,
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
        score_x_a,
        score_x_p,
        score_x_n,
        side_information_a,
        side_information_p,
        side_information_n,
        side_information_hat_a,
        side_information_hat_p,
        side_information_hat_n,
    ):
        reconstruction_loss_x_a = self.reconstruction_loss(x_a, reconstructed_x_a)
        reconstruction_loss_x_p = self.reconstruction_loss(x_p, reconstructed_x_p)
        reconstruction_loss_x_n = self.reconstruction_loss(x_n, reconstructed_x_n)

        if self.triplet_loss_objective == "latent_space":
            triplet_loss = self.triplet_loss(encoded_x_a, encoded_x_p, encoded_x_n)
            print('self.triplet_loss_objective == "latent_space"')
        elif self.triplet_loss_objective == "score":
            triplet_loss = self.triplet_loss(score_x_a, score_x_p, score_x_n)
            print('self.triplet_loss_objective == "score"')
        else:
            raise ValueError("triplet_loss_objective not in ['latent_space', 'score']")

        side_information_loss_x_a = self.side_information_loss(
            side_information_hat_a, torch.squeeze(side_information_a).long()
        )
        side_information_loss_x_p = self.side_information_loss(
            side_information_hat_p, torch.squeeze(side_information_p).long()
        )
        side_information_loss_x_n = self.side_information_loss(
            side_information_hat_n, torch.squeeze(side_information_n).long()
        )

        reconstruction_loss = (
            reconstruction_loss_x_a + reconstruction_loss_x_p + reconstruction_loss_x_n
        )
        side_information_loss = (
            side_information_loss_x_a
            + side_information_loss_x_p
            + side_information_loss_x_n
        )

        self.log("reconstruction_loss_x_a", reconstruction_loss_x_a, on_epoch=True)
        self.log("reconstruction_loss_x_p", reconstruction_loss_x_p, on_epoch=True)
        self.log("reconstruction_loss_x_n", reconstruction_loss_x_n, on_epoch=True)
        self.log("triplet_loss", triplet_loss, on_epoch=True)
        self.log("side_information_loss_x_a", side_information_loss_x_a, on_epoch=True)
        self.log("side_information_loss_x_p", side_information_loss_x_p, on_epoch=True)
        self.log("side_information_loss_x_n", side_information_loss_x_n, on_epoch=True)

        if self.uncertainty_loss:
            loss = self.multitaskloss_instance(
                torch.stack((reconstruction_loss, triplet_loss, side_information_loss))
            )
        else:

            loss = (
                self.alpha * reconstruction_loss
                + self.beta * triplet_loss
                + self.gamma * side_information_loss
            )

        return loss

    def get_score(self, x):
        return self.scorer(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_train_start(self):
        self.hparams.update(self.trainer.datamodule.get_summery())
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
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
            reconstructed_x_a,
            reconstructed_x_p,
            reconstructed_x_n,
        ) = self(x_a, x_p, x_n)

        score_x_a = self.get_score(encoded_x_a)
        score_x_p = self.get_score(encoded_x_p)
        score_x_n = self.get_score(encoded_x_n)

        side_information_hat_a = self.score_to_side_information(score_x_a)
        side_information_hat_p = self.score_to_side_information(score_x_p)
        side_information_hat_n = self.score_to_side_information(score_x_n)
        # side_information_hat_a, side_information_hat_p, side_information_hat_n = (
        #     None,
        #     None,
        #     None,
        # )

        loss = self.loss_function(
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
            side_information_hat_a,
            side_information_hat_p,
            side_information_hat_n,
        )
        self.log("training_loss", loss, on_epoch=True)
        return loss

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
            reconstructed_x_a,
            reconstructed_x_p,
            reconstructed_x_n,
        ) = self(x_a, x_p, x_n)

        score_x_a = self.get_score(encoded_x_a)

        return encoded_x_a, score_x_a

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
            reconstructed_x_a,
            reconstructed_x_p,
            reconstructed_x_n,
        ) = self(x_a, x_p, x_n)

        score_x_a = self.get_score(encoded_x_a)
        side_information_hat_a = self.score_to_side_information(score_x_a)

        side_information_loss = self.side_information_loss(
            side_information_hat_a, torch.squeeze(side_information_a).long()
        )

        y = y_a.squeeze().int()
        pred = score_x_a.argmax(dim=1)

        acc = FM.accuracy(pred, y)

        _, _, new_pred = match(y.numpy(), pred.numpy())
        aligned_acc = FM.accuracy(torch.tensor(new_pred), y)

        metrics = {
            "val_acc": acc,
            "val_aligned_acc": aligned_acc,
            "val_side_information_loss": side_information_loss,
        }
        self.log_dict(metrics, on_epoch=True)

        return metrics, x_a, side_information_a, encoded_x_a, score_x_a, y_a

    def validation_epoch_end(self, outputs):
        metrics = []
        x_as = []
        side_information_as = []
        encoded_x_as = []
        score_x_as = []
        y_as = []
        for out in outputs:
            metric, x_a, side_information_a, encoded_x_a, score_x_a, y_a = out
            metrics.append(metric)
            x_as.append(x_a)
            side_information_as.append(side_information_a)
            encoded_x_as.append(encoded_x_a)
            score_x_as.append(score_x_a)
            y_as.append(y_a)

        encoded_x_a = torch.cat(encoded_x_as)
        score_x_a = torch.cat(score_x_as)
        y_a = torch.cat(y_as).squeeze().int()

        score = score_x_a.argmax(dim=1)

        _, _, aligned_score = match(y_a.numpy(), score.numpy())
        aligned_score = torch.tensor(aligned_score)

        mapper = umap.UMAP().fit(encoded_x_a)
        self.logger.experiment.log(
            {
                "Score distribution": wandb.Histogram(score),
                "Aligned distribution": wandb.Histogram(aligned_score),
                "U-map Y": wandb.Image(umap.plot.points(mapper, y_a)),
                "U-map Y_hat": wandb.Image(
                    umap.plot.points(mapper, torch.squeeze(score))
                ),
                "U-map aligned Y_hat": wandb.Image(
                    umap.plot.points(mapper, torch.squeeze(aligned_score))
                ),
            }
        )

    def test_step(self, batch, batch_idx):
        (
            x_a,
            x_p,
            x_n,
            side_information_a,
            side_information_p,
            side_information_n,
        ) = batch

        (
            encoded_x_a,
            encoded_x_p,
            encoded_x_n,
            reconstructed_x_a,
            reconstructed_x_p,
            reconstructed_x_n,
        ) = self(x_a, x_p, x_n)

        score_x_a = self.get_score(encoded_x_a)

        loss = self.side_information_loss(
            score_x_a, torch.squeeze(side_information_a).long()
        )

        self.log("test_loss", loss)
