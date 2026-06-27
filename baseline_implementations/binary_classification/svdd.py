"""
Pytorch implementation of Deep SVDD:
    https://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf

Created on: 07/04/25
"""

import torch as T
import torch.nn as nn
from torch import Tensor
import numpy as np
from numpy import ndarray
from model.model import ContrastiveMLP
from typing import List, Callable, Optional
from baseline_implementations.common.process_batch import process_batch
from baseline_implementations.common.losses import BaseLoss
from baseline_implementations.common.model_eval import evaluation_function
from util.metrics import balanced_auroc


# -- loss function
def svdd_loss(
    z: Tensor,
    centroid: Tensor,
):
    return T.mean(T.sum(T.pow(z - centroid, 2), dim=-1))
    # return T.mean(T.sum(T.pow(z - centroid),2)), dim = 0)


class LossWrapper(BaseLoss):
    def forward(self, model, x, y, mixed_precision, training):
        logits, y_pred, y_true = self.get_model_input(model, x, y, mixed_precision)
        return self.loss(logits, model), self.calc_metric(
            logits.clone().detach(), y_true
        )


class _SVDDLoss(nn.Module):
    def forward(
        self,
        x: Tensor,
        model: nn.Module,
    ) -> Tensor:
        z = model(x)
        return svdd_loss(z, model.get_centroid())


def SVDDLoss(*args, **kwargs):
    return LossWrapper(_SVDDLoss, *args, **kwargs)


# -- model
def svdd_mlp(
    d_in: int,
    neurons: List[int],
    activation: Callable[[], nn.Module] = nn.ReLU,  # must not be sigmoid or tanh
    dropout: float = 0.0,
    residual: List[bool] = [],
    gated: bool = False,
    norm_layer: Callable[[int], nn.Module] = nn.Identity,
    dropout_layer: Callable[[int], nn.Module] = nn.Dropout,
    d_out: Optional[int] = None,
    layer_scale=None,
) -> ContrastiveMLP:

    return ContrastiveMLP(
        d_in=d_in,
        neurons=neurons,
        activation=activation,
        dropout=dropout,
        residual=residual,
        gated=gated,
        norm_layer=norm_layer,
        dropout_layer=dropout_layer,
        d_out=d_out,
        final_layer_activation=None,
        layer_scale=layer_scale,
        project_to_sphere=False,
        block_norm=False,
        bias=True,
    )


class SVDD(nn.Module):
    def __init__(
        self,
        d_in: int,
        neurons: List[int],
        activation: Callable[[], nn.Module] = nn.ReLU,  # must not be sigmoid or tanh
        dropout: float = 0.0,
        residual: List[bool] = [],
        gated: bool = False,
        norm_layer: Callable[[int], nn.Module] = nn.Identity,
        dropout_layer: Callable[[int], nn.Module] = nn.Dropout,
        d_out: Optional[int] = None,
        layer_scale=None,
        set_c_on_forward_pass: bool = True,
    ) -> None:
        super().__init__()

        self.d_out = d_out
        # self.register_buffer('c', T.empty(self.d_out).detach())
        self.register_buffer("c", (T.randn(self.d_out).detach() - 0.5) * 2e-6)
        self.set_c_on_forward_pass = set_c_on_forward_pass
        self.mlp = svdd_mlp(
            d_in=d_in,
            neurons=neurons,
            activation=activation,
            dropout=dropout,
            residual=residual,
            gated=gated,
            norm_layer=norm_layer,
            dropout_layer=dropout_layer,
            d_out=d_out,
            layer_scale=layer_scale,
        )

    def init_c(
        self,
        x_data: Optional[Tensor] = None,
        dataloader=None,
    ) -> None:

        with T.no_grad():
            if x_data is not None and dataloader is None:
                embeddings = self.mlp(x_data)

            elif dataloader is not None and x_data is None:
                device = next(self.parameters()).device
                embeddings = []
                for batch in dataloader:
                    x, _ = process_batch(
                        batch, device=device, mixed_precision=False, non_blocking=True
                    )
                    z = self.mlp(x)
                    embeddings.append(z)

                embeddings = T.cat(embeddings, dim=0)

            c = T.mean(embeddings, dim=0)
            if T.isnan(c).any() or T.isinf(c).any():
                raise ValueError("Centroid contains NaN or Inf")

            c = T.nan_to_num(c, nan=1e-6, posinf=1e6, neginf=-1e6)
            c[(T.abs(c) < 1e-6)] = 1e-6
            self.c.copy_(c.detach())

    def forward(self, x: Tensor) -> Tensor:
        if self.c.numel() == 0 and self.set_c_on_forward_pass:
            self.init_c(x_data=x)

        z = self.mlp(x)
        return z

    def reset_c(self) -> None:
        self.c.copy_(T.empty(self.d_out).detach())  # in reset_c

    def get_centroid(self) -> Tensor:
        return self.c


# -- evaluation function
@T.no_grad()
def get_anomaly_scores(
    model: nn.Module,
    centroid: Tensor,
    x_data: Optional[Tensor] = None,
    dataloader=None,
    chunk_size: Optional[int] = None,
) -> List[float]:

    device = next(model.parameters()).device
    # model.eval()

    if x_data is not None and dataloader is None:
        if chunk_size is None:
            z = model(x_data)
            scores = T.sum((z - centroid) ** 2, dim=-1).cpu().detach().numpy()
        else:
            num_samples = x_data.size(0)
            scores = []
            chunk_i = 0
            for idx in range(0, num_samples, chunk_size):
                chunk_i += 1
                x_chunk = x_data[idx : min((idx + chunk_size), num_samples), :]
                z_chunk = model(x_chunk)
                chunk_scores = (
                    T.sum((z_chunk - centroid) ** 2, dim=-1).cpu().detach().numpy()
                )
                scores.append(chunk_scores)
            scores = np.concatenate(scores, axis=0)

    elif dataloader is not None:
        scores = []
        for batch in dataloader:
            x, _ = process_batch(
                batch, device=device, mixed_precision=False, non_blocking=True
            )
            z = model(x)
            dist = T.sum((z - centroid) ** 2, dim=-1)
            scores.append(dist.cpu().detach().numpy())
        scores = np.concatenate(scores, axis=0)
    else:
        raise ValueError("Exactly one of x_data and dataloader must be provided.")
    return scores


@T.no_grad()
@evaluation_function
def svdd_engine(
    model: nn.Module,
    x_val: Tensor,
    y_val: ndarray,
    x_zd: Optional[Tensor] = None,
    y_zd: Optional[ndarray] = None,
    chunk_size: int = 1024,
    return_class_level: bool = False,
):
    print("IN SVDD EVAL FUNCTION")

    # concat zd and val data
    if x_zd is not None and y_zd is not None:
        x_val = T.cat((x_val, x_zd), dim=0)
        y_val = np.concatenate((y_val, y_zd), axis=0)

    val_dists = get_anomaly_scores(
        model=model,
        centroid=model.get_centroid(),
        x_data=x_val,
        chunk_size=chunk_size,
    )

    eval_scores = balanced_auroc(
        val_dists, y_val, return_class_level=return_class_level
    )

    print(eval_scores)

    if return_class_level:
        score_dict = {f"class_{i}_auroc": v for (i, v) in enumerate(eval_scores)}
        score_dict["mean"] = np.mean(eval_scores)
        return score_dict

    else:
        return dict(
            eval_score=eval_scores,
        )
