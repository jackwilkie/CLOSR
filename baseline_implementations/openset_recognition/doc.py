"""
Implementation of Deep Open Classification in Pytorch:

https://arxiv.org/abs/1709.08716

Created on: 06/06/24
"""

import torch as T
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from baseline_implementations.common.model_eval import evaluation_function
from baseline_implementations.common.losses import SupervisedLoss
from model.model import make_mlp
from typing import Optional
import numpy as np
from baseline_implementations.common.open_auc import open_auc


class DOC(nn.Module):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.mlp = make_mlp(*args, **kwargs)
        self.thresholds = None

    def forward_mlp(self, x):
        return self.mlp(x)

    def forward(self, x):
        x = self.forward_mlp(x)
        x = F.sigmoid(x)
        return x

    def fit_gaussian(self, x=None, y=None, dl=None, transform: bool = True):

        if dl is not None:
            # Calculate std using DataLoader in batches
            sum_squared_errors = None
            counts = None

            for batch_x, batch_y in dl:
                batch_x = self.forward(batch_x) if transform else batch_x

                if sum_squared_errors is None:
                    sum_squared_errors = T.zeros(
                        batch_x.size(-1), device=batch_x.device, dtype=batch_x.dtype
                    )

                if counts is None:
                    counts = T.zeros(
                        batch_x.size(-1), device=batch_x.device, dtype=batch_x.dtype
                    )

                # Assuming batch_x is the data and batch_y are the class labels
                _, batch_counts = T.unique(y, return_counts=True)
                counts = counts + batch_counts

                batch_errors = (batch_x - 1) * F.one_hot(
                    batch_y, num_classes=x.size(-1)
                )  # remove values not belonging to target class
                batch_errors = T.sum(T.pow(batch_errors, 2), 0)

                sum_squared_errors = sum_squared_errors + batch_errors

            stds = T.sqrt(sum_squared_errors / (counts + 1e-6))

        elif x is not None and y is not None:
            # Calculate std using entire dataset
            x = self.forward(x) if transform else x
            _, counts = T.unique(y, return_counts=True)
            stds = T.sqrt(
                T.sum(T.pow(((x - 1) * F.one_hot(y, num_classes=x.size(-1))), 2), dim=0)
                / (counts + 1e-6)
            )

        # use stds to calculate thresholds
        t = 1 - (3 * stds)
        t[t < 0.5] = 0.5
        self.stds = stds
        self.thresholds = t

    def forward_predict(self, x: Tensor) -> Tensor:
        x = self.forward(x)
        return self.predict(x)

    def forward_osr_eval(self, x: Tensor) -> Tensor:
        x = self.forward(x)
        osr_scores = self.get_osr_scores(x)
        osr_scores = x / (self.stds + 1e-6)
        osr_scores = T.max(osr_scores, dim=-1)[0]
        osr_scores = osr_scores  # higher score the more likely the sample is from unkown distribution
        return x, osr_scores

    def get_osr_scores(self, probs: Tensor) -> Tensor:
        osr_scores = probs / (self.stds + 1e-6)
        osr_scores = T.max(osr_scores, dim=-1)[0]
        return -1 * osr_scores

    def predict(self, x: Tensor) -> Tensor:
        if self.thresholds is None:
            raise ValueError(
                "ERROR::: Guassian filter must be fitted before making predictions"
            )

        below_threshold = x < self.thresholds
        all_below = below_threshold.all(dim=1)  # get zero day attack predictions
        y_pred = T.argmax(x, dim=-1)  # get known class predictions
        y_pred = T.where(
            all_below, x.size(1), y_pred
        )  # replace known labels where sample is predicted to be zero day
        return y_pred


class DOCLoss_(nn.Module):
    def forward(self, x, y):
        # convert labels to one hot
        if y.dim() == x.dim() - 1:
            # convert integer labels to one hot
            y = F.one_hot(y, num_classes=x.size(-1)).to(dtype=x.dtype, device=x.device)
        elif y.dim() != x.dim():
            # labels not ints or one hot
            raise ValueError("ERROR:: Invalid Label Shape")

        return F.binary_cross_entropy(x, y.float())


def DOCLoss(*args, **kwargs):
    return SupervisedLoss(
        *args,
        loss=DOCLoss_,
        **kwargs,
    )


@T.no_grad()
@evaluation_function
def doc_evaluation(
    model: nn.Module,
    x_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    y_val: Tensor,
    x_zd: Optional[Tensor] = None,
    y_zd: Optional[Tensor] = None,
) -> dict:

    # -- append zero day data to validation data
    if x_zd is not None:
        x_val = T.cat((x_val, x_zd), dim=0)
        y_val = np.concatenate((y_val, y_zd), axis=0)

    # -- get data logits
    train_embeddings = model(x_train)
    val_embeddings = model(x_val)

    # -- fit gaussian on training data
    model.fit_gaussian(train_embeddings, y_train, transform=False)

    # -- evaluate validation data
    osr_scores = model.get_osr_scores(val_embeddings).cpu().detach().numpy()
    val_embeddings = val_embeddings.cpu().detach().numpy()

    # -- calculate metrics
    score = open_auc(
        probs=val_embeddings, osr_scores=osr_scores, y_true=y_val, balanced=False
    )  # calculate OpenAUC score

    balanced_score = open_auc(
        probs=val_embeddings, osr_scores=osr_scores, y_true=y_val, balanced=True
    )  # calculate balanced OpenAUC score

    return {
        "OpenAUC": score,
        "balanced_OpenAUC": balanced_score,
    }  # return results as dict
