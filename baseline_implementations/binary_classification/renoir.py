"""
RENOIR Implementation in pytorch
"""

import torch as T
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from model.model import make_mlp
from typing import List, Type
from baseline_implementations.common.losses import BaseLoss
from .autoencoder import MSELoss, AutoEncoder
from baseline_implementations.common.training_loops import train
from baseline_implementations.common.model_eval import model_eval, evaluation_function
from baseline_implementations.common.process_batch import extract_features
from util.metrics import mean_auroc
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def _make_dataloader(x, y, device, config, shuffle=True):
    dataset_config = config.get("dataset", {}) if config is not None else {}
    batch_size = dataset_config.get("batch_size", 256)
    drop_last = dataset_config.get("drop_last", False)

    if x is None or y is None:
        return None

    if not isinstance(x, Tensor):
        x = T.tensor(x, dtype=T.float32)
    if not isinstance(y, Tensor):
        y = T.tensor(y, dtype=T.int64)

    dataset = TensorDataset(x.to(device), y.to(device))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )


def init_training(
    model,
    x_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    y_val: Tensor,
    device: str,
    config: dict,
    optimiser=None,
    scalar=None,
    schedule=None,
):
    train_dl = _make_dataloader(x_train, y_train, device, config, shuffle=True)
    val_dl = _make_dataloader(x_val, y_val, device, config, shuffle=False)

    if optimiser is None or schedule is None:
        raise ValueError(
            "init_training requires optimiser and schedule arguments. "
            "Create them in the calling training script."
        )

    return train_dl, val_dl, optimiser, scalar, schedule


class RenoirAutoEncoder(AutoEncoder):
    def __init__(
        self,
        d_in: int,
        dropout: float = 0.0,
        nuerons: List[int] = [32, 16, 32],
    ) -> None:
        super().__init__(
            d_in=d_in,
            neurons=[32, 16, 32],
            dropout=dropout,
        )


class Renoir(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        mlp_neurons: List[int],
        mlp_activation: Type[nn.Module] = nn.ReLU,
        mlp_final_activation: Type[nn.Module] = nn.Sigmoid,
        mlp_dropout: float = 0.0,
        autoencoder_dropout: Optional[float] = None,
    ):
        super().__init__()

        autoencoder_dropout = autoencoder_dropout or mlp_dropout

        self.mlp = make_mlp(
            d_in=d_in,
            d_out=d_out,
            neurons=mlp_neurons,
            activation=mlp_activation,
            dropout=mlp_dropout,
            final_layer_activation=mlp_final_activation,
        )

        self.attack_encoder = RenoirAutoEncoder(d_in=d_in, dropout=autoencoder_dropout)

        self.benign_encoder = RenoirAutoEncoder(
            d_in=d_in,
            dropout=autoencoder_dropout,
        )

    def forward_attack(self, x: Tensor) -> Tensor:
        # get reconstruction using attack encoder
        return self.attack_encoder(x)

    def forward_benign(self, x: Tensor) -> Tensor:
        # get reconstruction using benign encoder
        return self.benign_encoder(x)

    def forward_mlp(self, x: Tensor) -> Tensor:
        # pass through mlp
        return self.mlp(x)

    def forward_reconstruction(self, x: Tensor) -> Tensor:
        # get data reconstructions
        xbar_attack = self.forward_attack(x)
        xbar_benign = self.forward_benign(x)
        return xbar_benign, xbar_attack

    def forward_representation(self, x: Tensor) -> Tensor:
        # get data reconstructions
        xbar_benign, xbar_attack = self.forward_reconstruction(x)

        # pass original data aswell as reconstruciton through mlp
        z = self.mlp(x)
        z_attack = self.mlp(xbar_attack)
        z_benign = self.mlp(xbar_benign)

        return z, z_benign, z_attack

    def forward(self, x: Tensor) -> Tensor:
        z, z_benign, z_attack = self.forward_representation(x)

        # calculate euclidean distance between original data and reconstructions
        d_ab = F.pairwise_distance(z, z_benign)
        d_aa = F.pairwise_distance(z, z_attack)
        return d_aa - d_ab  # will return negative if a sample is an attack


class RenoirMSELoss(MSELoss):
    def forward(self, model, x, y, mixed_precision, training):
        x, y, y_true = self.get_model_input(model, x, y, mixed_precision)
        y[y > 1] = 1
        y_true[y_true > 1] = 1
        x_benign, x_attack = None, None

        if 0 in y_true:
            x_benign = x[y_true == 0]
            xbar_benign = model.forward_benign(x_benign)

        if any(y_true > 0):
            x_attack = x[y_true != 0]
            xbar_attack = model.forward_attack(x_attack)

        attack_loss = self.loss(xbar_attack, x_attack) if x_attack is not None else 0.0
        benign_loss = self.loss(xbar_benign, x_benign) if x_benign is not None else 0.0
        loss = attack_loss + benign_loss

        return loss, self.calc_metric(x.clone().detach(), y_true)


class RenoirTripletLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(
        self, x: Tensor, xbar_benign: Tensor, xbar_attack: Tensor, y: Tensor
    ) -> Tensor:
        d_xa = F.pairwise_distance(
            x, xbar_attack
        )  # distance from anchor to all attack reconstructions
        d_xb = F.pairwise_distance(
            x, xbar_benign
        )  # distance from anchor to all benign encoder reconstructions

        d_ap = T.where(
            y == 0, d_xb, d_xa
        )  # distance between anchor and similar samples
        d_an = T.where(
            y == 0, d_xa, d_xb
        )  # distance between anchor and dissimilar sapmles

        """
        calculate soft margin triplet loss using:
            loss = ln(1 + exp(d_ap^2 - d_an^2))
        """
        loss = T.log(1 + T.exp(T.pow(d_ap, 2) - T.pow(d_an, 2)))
        return T.mean(loss)


class RenoirLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            loss=RenoirTripletLoss,
            **kwargs,
        )

    def forward(self, model, x, y, mixed_precision, training):
        x, y, y_true = self.get_model_input(model, x, y, mixed_precision)
        y[y > 1] = 1
        y_true[y_true > 1] = 1
        xbar_benign, xbar_attack = model.forward_reconstruction(x)

        # pass through mlp
        x_a = model.forward_mlp(x)
        xbar_benign = model.forward_mlp(xbar_benign.detach())
        xbar_attack = model.forward_mlp(xbar_attack.detach())
        loss = self.loss(
            x=x_a, xbar_benign=xbar_benign, xbar_attack=xbar_attack, y=y_true
        )

        # print(loss)
        return loss, self.calc_metric(model(x), y_true.clone().detach())

    def calc_metric(self, x, y_true):
        # get predictions
        y_pred = x < 0
        correct = y_pred.bool() == y_true.bool()
        correct = correct.float()
        correct = correct.sum()
        acc = correct / y_true.size(0)
        return acc.cpu().detach().numpy()


def train_renoir_(
    model,
    mlp_train_dl,
    mlp_val_dl,
    mlp_epochs: int,
    mlp_optimiser,
    mlp_schedule,
    mlp_loss=RenoirLoss(),
    autoencoder_epochs: Optional[int] = None,
    autoencoder_optimiser=None,
    autoencoder_schedule=None,
    autoencoder_loss=RenoirMSELoss(),
    ep_log_interval: int = 1,
    logger=None,
    scalar=None,
    grad_accumulation=1,
    start_epoch=0,
    rank=None,
    world_size=None,
    autoencoder_train_dl=None,
    autoencoder_val_dl=None,
):
    # initalise autoencoder trianing
    autoencoder_epochs = autoencoder_epochs or mlp_epochs
    autoencoder_optimiser = autoencoder_optimiser or mlp_optimiser
    autoencoder_schedule = autoencoder_schedule or mlp_schedule
    autoencoder_train_dl = autoencoder_train_dl or mlp_train_dl
    autoencoder_val_dl = autoencoder_val_dl or mlp_val_dl

    model.attack_encoder.requires_grad = True
    model.benign_encoder.requires_grad = True

    # train autoencoder
    train(
        model=model,
        optimiser=autoencoder_optimiser,
        loss_calc=autoencoder_loss,
        epochs=autoencoder_epochs,
        train_dl=autoencoder_train_dl,
        val_dl=autoencoder_val_dl,
        ep_log_interval=ep_log_interval,
        scheduler=autoencoder_schedule,
        logger=logger,
        scalar=scalar,
        print_batch_loss=False,
        max_batches=0,
        grad_accumulation=grad_accumulation,
        start_epoch=start_epoch,
        rank=rank,
        world_size=world_size,
        val_pass=False,
    )

    model.attack_encoder.requires_grad = False
    model.benign_encoder.requires_grad = False
    mlp_schedule.reset()

    # train mlp
    train(
        model=model,
        optimiser=mlp_optimiser,
        loss_calc=mlp_loss,
        epochs=mlp_epochs,
        train_dl=mlp_train_dl,
        val_dl=mlp_val_dl,
        ep_log_interval=ep_log_interval,
        scheduler=mlp_schedule,
        logger=logger,
        scalar=scalar,
        print_batch_loss=False,
        max_batches=0,
        grad_accumulation=grad_accumulation,
        start_epoch=start_epoch,
        rank=rank,
        world_size=world_size,
        val_pass=False,
    )

    return None


def train_renoir(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    device,
    config,
    optimiser=None,
    scalar=None,
    schedule=None,
    logger=None,
    return_features=True,
):
    train_dl, val_dl, optimiser, scalar, schedule = init_training(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        device,
        config,
        optimiser=optimiser,
        scalar=scalar,
        schedule=schedule,
    )
    _ = train_renoir_(
        model,
        train_dl,
        val_dl,
        mlp_epochs=config["hyperparameters"]["training_epochs"],
        mlp_optimiser=optimiser,
        mlp_schedule=schedule,
        ep_log_interval=1,
        logger=logger,
        scalar=scalar,
        grad_accumulation=config["hyperparameters"]["grad_accumulation"],
    )

    # get ouptput logits of fully trained model
    if return_features:
        with T.no_grad():
            z_train, y_train = extract_features(
                model=model,
                dl=train_dl,
                mixed_precision=False,
                non_blocking=True,
                batch=False,
            )

            if val_dl is not None:
                z_val, y_val = extract_features(
                    model=model,
                    dl=val_dl,
                    mixed_precision=False,
                    non_blocking=True,
                    batch=False,
                )
            else:
                z_train = z_val = None
        return z_train, z_val


@evaluation_function
def eval_renoir(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
):
    y_train[y_train > 1] = 1
    y_test[y_test > 1] = 1

    train_logits = model(x_train)
    y_pred_train = train_logits < 0
    y_pred_train = y_pred_train.to(T.int64).cpu().detach().numpy()
    train_results = model_eval(
        y_train,
        y_pred_train,
        label="train",
        return_class_level=True,
        return_detection_metrics=True,
    )

    test_logits = model(x_test)
    y_pred_test = test_logits < 0
    y_pred_test = y_pred_test.to(T.int64).cpu().detach().numpy()
    test_results = model_eval(
        y_test,
        y_pred_test,
        label="test",
        return_class_level=True,
        return_detection_metrics=True,
    )

    return {**train_results, **test_results}


@evaluation_function
def eval_renoir_auroc(
    model,
    x_test,
    y_test,
    return_class_level: bool = False,
    label_prefix: Optional[str] = None,
):
    # -- preprocess arguments
    label_prefix = label_prefix or ""

    device = next(model.parameters()).device

    if not isinstance(x_test, Tensor):
        x_test = T.tensor(x_test, dtype=T.float32, device=device)

    test_logits = model(x_test)
    y_pred_test = test_logits < 0
    y_pred_test = y_pred_test.to(T.int64).cpu().detach().numpy()
    class_auroc = mean_auroc(
        scores=y_pred_test, y_true=y_test, return_class_level=return_class_level
    )

    if return_class_level:
        test_results = {
            f"{label_prefix}test_class_{i + 1}_auroc": score
            for i, score in enumerate(class_auroc)
        }
        test_results[f"{label_prefix}test_mean_auroc"] = np.mean(class_auroc)
    else:
        test_results = {f"{label_prefix}test_mean_auroc": class_auroc}

    return test_results
