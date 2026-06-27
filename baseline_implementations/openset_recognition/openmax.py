"""
Implementation of Openmax in pytorch:

    https://arxiv.org/pdf/1511.06233

Created on: 11/06/24
"""

import torch as T
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from functools import partial
import libmr

from baseline_implementations.common.model_eval import model_eval, evaluation_function
from model.model import make_mlp
from baseline_implementations.common.metrics import euclidean_distance, cosine_distance
from baseline_implementations.common.open_auc import open_auc
from typing import Optional


def eucos_dist(a: Tensor, b: Tensor, weight=1.0) -> Tensor:
    return (weight * euclidean_distance(a, b)) + cosine_distance(a, b)


class OSRModel(nn.Module):
    def __init__(
        self,
        eucos_weight: float = 1.0,
        distance_metric: str = "euclidean",
    ):
        super().__init__()
        self.weibulls = None
        self.centroids = None
        self.distance_metric = euclidean_distance
        # get distance metric
        self.eucos_weight = eucos_weight  # weight for eucos distance metric

        if distance_metric == "euclidean":
            self.distance_metric = euclidean_distance

        elif distance_metric == "cosine":
            self.distance_metric = cosine_distance

        elif distance_metric == "eucos":
            self.distance_metric = partial(eucos_dist, weight=eucos_weight)

        else:
            raise ValueError(f"Invalid distance metric provided, got {distance_metric}")

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

    @T.no_grad()
    def fit_weibull(
        self,
        x: Tensor = None,
        y: Tensor = None,
        dl=None,
        nu_val: int = 20,  # number of values to use to fit weibull
        extract_features: bool = False,
    ) -> None:

        if extract_features:
            z_fit, y_true = extract_features(
                model=self.mlp,
                x_data=x,
                y_data=y,
                dl=dl,
                batch=True,
            )  # get predictions and truelabels
        else:
            # assumes mavs are provided already if extract features is false
            z_fit, y_true = x, y

        y_pred = T.argmax(z_fit, dim=1)  # get predicted labels

        # calculate class means based on long tail of correct classifications
        # size is C x C
        n_classes = z_fit.size(-1)
        self.centroids = T.zeros(
            (n_classes, n_classes), dtype=z_fit.dtype, device=z_fit.device
        )  # init centroids
        self.weibulls = []

        # calculate centroid for each class
        for c in range(n_classes):
            class_samples = z_fit[
                (y_true == c) & (y_pred == c)
            ]  # get correctly classified samples

            if class_samples.size(0) > 0:  # avoid zero division
                self.centroids[c] = T.mean(class_samples, dim=0)  # calculate mean

                # calculate class distances from centroids
                class_dists = self.distance_metric(class_samples, self.centroids[c])
                top_values, _ = T.topk(
                    class_dists, min(nu_val, class_dists.size(0))
                )  # get nu highest distances

                # fit weibull distributino on nu highest distances
                distribution = libmr.MR()
                distribution.fit_high(top_values, len(top_values))
                self.weibulls.append(distribution)

            else:
                print(f"WARNING:::: No samples give for class {c}")
                self.weibulls.append(None)

                # raise ValueError(f'ERROR::: No samples give for class {c}')

    def predict_label(self, z: Tensor, eps: float = 0.0) -> Tensor:
        z = F.softmax(z)
        max_probs, _ = T.max(z, dim=1, keepdim=True)
        y_pred = T.where(max_probs < eps, T.tensor(0), T.argmax(z, dim=1))
        y_pred = y_pred - 1  # unknown classes given label -1
        return y_pred

    @T.no_grad()
    def update_probs(self, z, mavs=None, alpha: int = None):

        B = z.size(0) if z.dim() > 1 else 1  # get batch size
        z = z.unsqueeze(0) if z.dim() == 1 else z

        n_classes = z.size(-1)
        alpha = alpha or n_classes
        predictions = T.zeros((B, n_classes + 1))
        predictions[:, :n_classes] = z.clone()

        mavs = mavs if mavs is not None else z
        mav_dists = [self.distance_metric(mavs, c) for c in self.centroids]

        # iterate over samples
        for i_sample, sample in enumerate(z):
            # get highest class probabilities
            top_values, top_indices = T.topk(sample, alpha)
            unkown_prob = 0.0
            mav = mavs[i_sample] if mavs is not None else sample

            for i_adjust, i_feature in enumerate(top_indices):
                # mav_dist = self.distance_metric(mav, self.centroids[i_feature])
                mav_dist = mav_dists[i_feature][i_sample]

                if self.weibulls[i_feature] is not None:
                    a = (alpha - i_adjust) / alpha
                    w_val = 1 - (a * self.weibulls[i_feature].w_score(mav_dist))

                    predictions[i_sample, i_feature] *= (
                        w_val  # scale by chance of belonging to distribution
                    )
                    unkown_prob += mav[i_feature] * (
                        1 - w_val
                    )  # get prob of not being in distribution

            predictions[i_sample, n_classes] = unkown_prob

        return F.softmax(predictions, dim=-1)


class OpenMax(OSRModel):
    def __init__(
        self,
        *args,
        distance_metric: str = "euclidean",
        eucos_weight: float = 1.0,
        **kwargs,
    ) -> None:

        super().__init__()
        self.mlp = make_mlp(*args, **kwargs)
        self.weibulls = None
        self.centroids = None

        # get distance metric
        self.eucos_weight = eucos_weight  # weight for eucos distance metric

        if distance_metric == "euclidean":
            self.distance_metric = euclidean_distance

        elif distance_metric == "cosine":
            self.distance_metric = cosine_distance

        elif distance_metric == "eucos":
            self.distance_metric = partial(eucos_dist, weight=eucos_weight)

        else:
            raise ValueError(f"Invalid distance metric provided, got {distance_metric}")

    def forward_openmax(self, x: Tensor, alpha: Optional[int] = None) -> Tensor:
        return self.update_probs(self.mlp(x), alpha=alpha)


@evaluation_function
def openmax_eval(
    model,
    x,
    y,
    n_known_classes: int = None,
    label_prefix="",
):
    y_pred = model.predict_label(model.forward_openmax(x))

    # combine zero day classes
    # precision doesnt make sense otherwise so must evaluate accuracy seperately
    zd_results = {}
    if n_known_classes is not None:
        # Identify unknown classes
        unknown_mask = y >= n_known_classes
        unique_unknown_classes = T.unique(y[unknown_mask])

        # Calculate accuracy for each unknown class individually
        for cls in unique_unknown_classes:
            cls_mask = y == cls
            # Check if any predictions correspond to this class and calculate accuracy
            cls_pred_correct = (y_pred[cls_mask] == -1).float().mean()
            zd_results[f"{label_prefix}zd_acc_class_{cls.item()}"] = (
                cls_pred_correct.item()
            )

        y[y >= n_known_classes] = n_known_classes
        y_pred[y_pred == -1] = n_known_classes

    results = model_eval(
        y_pred,
        y,
        return_class_level=True,
        return_detection_metrics=True,
        lablel=label_prefix,
    )

    return {**results, **zd_results}


@T.no_grad()
@evaluation_function
def openmax_engine(
    model,
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    alpha: float = 10,
    nu: int = 20,
    x_zd=None,
    y_zd=None,
):

    # -- get train and test mavs
    if x_zd is not None:
        x_test = T.cat((x_test, x_zd), dim=0)
        y_test = np.concatenate((y_test, y_zd), axis=0)

    x_train = model(x_train)
    x_test = model(x_test)

    # -- fit openmax weibull distributions
    model.fit_weibull(
        x=x_train,
        y=y_train,
        nu_val=nu,
    )

    osr_logits = model.update_probs(
        z=x_test,
        alpha=alpha,
    )
    osr_scores = osr_logits[:, -1]

    # -- evaluate openAUC scores
    x_test = x_test.cpu().detach().numpy()
    osr_scores = osr_scores.cpu().detach().numpy()

    score = open_auc(
        probs=x_test,
        osr_scores=osr_scores,
        y_true=y_test,
        balanced=False,
    )  # calculate OpenAUC score

    balanced_score = open_auc(
        probs=x_test,
        osr_scores=osr_scores,
        y_true=y_test,
        balanced=True,
    )  # calculate OpenAUC score

    return {
        "alpha": alpha,
        "nu": nu,
        "OpenAuc": score,
        "balanced_OpenAuc": balanced_score,
    }
