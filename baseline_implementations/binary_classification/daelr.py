"""
Pytorch implementations of DAR-LR and DUAD-LR
    https://arxiv.org/html/2407.08838v2

Created on: 08/04/25
"""

import torch as T
from torch import Tensor
import torch.nn as nn
from . import autoencoder
from typing import Type, List, Optional
from baseline_implementations.common.losses import BaseLoss
import numpy as np
from numpy import ndarray
from baseline_implementations.common.model_eval import evaluation_function
from util.metrics import balanced_auroc
from baseline_implementations.common.process_batch import process_batch


# -- define loss
def calc_mse(
    x,
    x_true,
):
    return T.mean((x - x_true) ** 2, dim=-1)


def daelr_loss(
    x: Tensor,
    x_bar: Tensor,
    z: Tensor,
    c: Tensor,
    lam: float,
) -> Tensor:
    return T.mean(calc_mse(x, x_bar) + (lam * calc_mse(z, c)))


class LossWrapper(BaseLoss):
    def forward(self, model, x, y, mixed_precision, training):
        x, _, y_true = self.get_model_input(model, x, y, mixed_precision)
        z = model.encode(x)
        x_bar = model.decode(z)
        return self.loss(x, x_bar, z, model.get_c()), self.calc_metric(
            x.clone().detach(), y_true
        )


class _DAELRLoss(nn.Module):
    def __init__(
        self,
        lam: float,
    ):
        super().__init__()
        self.lam = lam

    def forward(
        self,
        x: Tensor,
        x_bar: Tensor,
        z: Tensor,
        c: Tensor,
    ) -> Tensor:
        return daelr_loss(x=x, x_bar=x_bar, z=z, c=c, lam=self.lam)


def DAELRLoss(*args, **kwargs):
    return LossWrapper(_DAELRLoss, *args, **kwargs)


# --- define model
class DAELR(autoencoder.SymetricAutoEncoder):
    """
    Autoencoder with additional clustering regularisation term
    """

    def __init__(
        self,
        d_in: int,
        neurons: List[int],
        bottleneck_ratio: float,
        dropout: float = 0.0,
        dropout_layer: Type[nn.Module] = nn.Dropout,
        final_layer_activation: Optional[Type[nn.Module]] = None,
    ):
        super().__init__(
            d_in=d_in,
            neurons=neurons,
            bottleneck_ratio=bottleneck_ratio,
            dropout=dropout,
            dropout_layer=dropout_layer,
            final_layer_activation=final_layer_activation,
        )

        self.c = nn.Parameter(T.randn(int(bottleneck_ratio * d_in)) * 1e-6)

    def get_c(self) -> Tensor:
        return self.c


# -- engine


# -- evaluation function
@T.no_grad()
def get_anomaly_scores(
    model: nn.Module,
    x_data: Optional[Tensor] = None,
    dataloader=None,
    chunk_size: Optional[int] = None,
) -> List[float]:

    device = next(model.parameters()).device

    if x_data is not None and dataloader is None:
        if chunk_size is None:
            z = model(x_data)
            scores = T.sum((z - x_data) ** 2, dim=-1).cpu().detach().numpy()
        else:
            num_samples = x_data.size(0)
            scores = []
            chunk_i = 0
            for idx in range(0, num_samples, chunk_size):
                chunk_i += 1
                x_chunk = x_data[idx : min((idx + chunk_size), num_samples), :]
                z_chunk = model(x_chunk)
                chunk_scores = (
                    T.sum((z_chunk - x_chunk) ** 2, dim=-1).cpu().detach().numpy()
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
            score = T.sum((z - x) ** 2, dim=-1).cpu().detach().numpy()
            scores.append(score)
        scores = np.concatenate(scores, axis=0)
    else:
        raise ValueError("Exactly one of x_data and dataloader must be provided.")
    return scores


@T.no_grad()
@evaluation_function
def daelr_engine(
    model: nn.Module,
    x_val: Tensor,
    y_val: ndarray,
    x_zd: Optional[Tensor] = None,
    y_zd: Optional[ndarray] = None,
    chunk_size: int = 1024,
    return_class_level: bool = False,
):
    # concat zd and val data
    if x_zd is not None and y_zd is not None:
        x_val = T.cat((x_val, x_zd), dim=0)
        y_val = np.concatenate((y_val, y_zd), axis=0)

    val_scores = get_anomaly_scores(
        model=model,
        x_data=x_val,
        chunk_size=chunk_size,
    )

    eval_scores = balanced_auroc(
        val_scores, y_val, return_class_level=return_class_level
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
