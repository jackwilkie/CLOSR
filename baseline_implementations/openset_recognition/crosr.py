"""
CROSR Implementation in pytorch:

    https://arxiv.org/pdf/1812.04246

Created on: 12/06/24
"""

# NOTES::: SET TAIL SIZE TO 50
# USES CLASS THRESEHOLD OF 0.5
# alpha is 10

import torch as T
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Callable, Tuple
import libmr
import numpy as np

from baseline_implementations.common.open_auc import open_auc
from baseline_implementations.common.process_batch import process_batch
from baseline_implementations.common.model_eval import model_eval, evaluation_function
from model.model import DenseBlock
from baseline_implementations.common.losses import SupervisedLoss
from .openmax import OSRModel


class DHRN(OSRModel):
    """
    Deep hierachical reconstruction net.
    MLP where the latent from each layer is passed through a bottleneck and then reconstructed
    to encourage encoding semantic information in the bottlenecks
    """

    def __init__(
        self,
        *args,
        d_in: int,
        dropout: float = 0.0,
        neurons: Optional[Union[List[int], int]] = None,
        n_layers: Optional[int] = None,
        bottleneck_ratio: Optional[float] = None,
        bottlesize: Optional[Union[List[int], int]] = None,
        n_classes: Optional[int] = None,
        act_fn: Callable[..., nn.Module] = nn.ReLU,
        dropout_layer: Callable[[float], nn.Module] = nn.Dropout,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # -- initialise weibulls for open set recognition
        self.d_in = d_in
        self.weibulls = None
        self.centroids = None

        # -- initalise number of layers and layer sizes in network
        if isinstance(neurons, int) and n_layers is not None:
            self.neurons = [neurons for _ in n_layers]

        elif (
            isinstance(neurons, list) or isinstance(neurons, int)
        ) and n_layers is None:
            neurons = [neurons] if isinstance(neurons, int) else neurons
            self.neurons = neurons

        else:
            raise ValueError(
                "ERROR::: Invalid combination of nuerons and n_layers arguments"
            )

        self.neurons = [d_in] + self.neurons

        # -- initalise bottleneck layer sizes
        if isinstance(bottleneck_ratio, float) and bottlesize is None:
            self.bottle_sizes = [
                int(bottleneck_ratio * n) for n in self.neurons[1:]
            ]  # consistent bottleration across layers

        elif bottleneck_ratio is None and isinstance(bottlesize, int):
            self.bottle_sizes = [
                bottlesize for _ in self.neurons[1:]
            ]  # consistant bottlesize across layers

        elif bottleneck_ratio is None and isinstance(bottlesize, list):
            self.bottle_sizes = bottlesize  # custom bottlesize for each layer

        else:
            raise ValueError(
                "ERROR::: Invalid combination of bottle ratio and bottlesize arguments"
            )

        # -- create neural network
        self.mlp_layers = nn.ModuleList(
            [
                DenseBlock(
                    in_dim=self.neurons[i],
                    out_dim=self.neurons[i + 1],
                    dropout=dropout,
                    dropout_layer=dropout_layer,
                    activation=act_fn,
                    residual=False,
                )
                for i in range(len(self.neurons) - 1)
            ]
        )

        self.bottlenecks = nn.ModuleList(
            [
                DenseBlock(
                    in_dim=n_size,
                    out_dim=b_size,
                    dropout=dropout,
                    dropout_layer=dropout_layer,
                    activation=act_fn,
                    residual=False,
                )
                for n_size, b_size in zip(self.neurons[1:], self.bottle_sizes)
            ]
        )

        self.reconstruction = nn.ModuleList(
            [
                DenseBlock(
                    in_dim=b_size,
                    out_dim=d_in,
                    dropout=dropout,
                    dropout_layer=dropout_layer,
                    activation=nn.Identity,
                    residual=False,
                )
                for b_size in self.bottle_sizes
            ]
        )
        self.reconstruction = self.reconstruction[::-1]

        self.ladder_block = nn.ModuleList(
            [
                DenseBlock(
                    in_dim=d_in,
                    out_dim=d_in,
                    dropout=dropout,
                    dropout_layer=dropout_layer,
                    activation=act_fn,
                    residual=False,
                )
                for _ in range(len(self.neurons) - 1)
            ]
        )

        # -- add cls head if number of classes specified
        self.latent_size = sum(self.bottle_sizes)
        self.cls_head = (
            nn.Linear(self.neurons[-1], n_classes)
            if n_classes is not None
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        """
        forward pass of network
        returns
        """
        # -- expand data to 2d if required
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, F = x.size()  # input is 2D, batch dim and feature dim

        # -- perform forward direction pass of ladder network
        latents = []

        for mlp, bottleneck in zip(self.mlp_layers, self.bottlenecks):
            x = mlp(x)  # pass through mlp
            latents.append(bottleneck(x.clone()))  # store latents

        # -- go backwards down ladder network to get reconstructed representation
        latents = latents[::-1]  # reverse order of latents
        x_bar = T.zeros(
            self.d_in, dtype=latents[0].dtype, device=latents[0].device
        )  # initalise reconstruction

        for latent, recon_layer, ladder_layer in zip(
            latents, self.reconstruction, self.ladder_block
        ):
            r = recon_layer(latent)
            x_bar = ladder_layer(x_bar + r)

        # -- remove batch if not in input
        if squeeze:
            x = x.squeeze(0)

        # -- return class predictions, concatted latents, and input reconstruction
        x = self.cls_head(x)  # get class predictions from logits
        latents = T.cat(latents, dim=-1)  # concat latents into single tensor

        return x, latents, x_bar

    def forward_mlp(self, x: Tensor) -> Tensor:
        for mlp in self.mlp_layers:
            x = mlp(x)  # pass through mlp
        return x

    def get_latents(self, x: Tensor, chunk_size: int = 1024) -> Tuple[Tensor, Tensor]:
        logits, latents = [], []
        n_samples = x.size(0)
        chunk_i = 0

        for idx in range(0, n_samples, chunk_size):
            chunk_i += 1

            # chunk features to compare to all test samples
            x_chunk = x[
                idx : min((idx + chunk_size), n_samples), :
            ]  # use rest of test data if not enough for chun
            logits_chunk, latents_chunk, _ = self.forward(x_chunk)
            logits.append(logits_chunk.detach())
            latents.append(latents_chunk.detach())

        logits = T.cat(logits, dim=0)
        latents = T.cat(latents, dim=0)

        return logits, latents

    @T.no_grad()
    def fit_weibull(
        self,
        x: Tensor = None,
        y: Tensor = None,
        latents: Tensor = None,
        dl=None,
        nu_val: int = 50,  # number of values to use to fit weibull
        extract_features: bool = False,
        chunk_size: Optional[int] = None,
    ) -> None:

        if extract_features:
            if x is not None and y is not None:
                z, latents, _ = self.forward(x)
                y_true = y

            elif dl is not None and x is None and y is None:
                z = []
                latents = []
                y_true = []

                for batch in dl:
                    x, y_batch = process_batch(
                        batch,
                        device=next(self.parameters()).device,
                        mixed_precision=False,
                        non_blocking=True,
                    )
                    z_batch, latents_batch, _ = self.forward(x)

                    # FIXME potentially move to cpu
                    y_true.append(y_batch)
                    z.append(z_batch)
                    latents.append(latents_batch)

                # concat lists into tensors
                y = T.cat(y, dim=0)
                z = T.cat(z, dim=0)
                latents = T.cat(latents, dim=0)

            else:
                raise ValueError("ERROR::: Invalid combinations of inputs recieved")
        else:
            z = x

        y_pred = T.argmax(z, dim=1)  # get predicted labels

        # calculate class means based on long tail of correct classifications
        # size is C x C
        n_classes = z.size(-1)
        latent_dims = latents.size(-1) if latents is not None else z.size(-1)
        self.centroids = T.zeros(
            (n_classes, latent_dims), dtype=z.dtype, device=z.device
        )  # init centroids
        self.weibulls = []

        # calculate centroid for each class
        for c in range(n_classes):
            class_samples = latents[
                (y == c) & (y_pred == c)
            ]  # get correctly classified samples

            if class_samples.size(0) > 0:  # avoid zero division
                self.centroids[c] = T.mean(class_samples, dim=0)  # calculate mean
                # calculate class distances from centroids
                class_dists = self.distance_metric(
                    class_samples, self.centroids[c], chunk_size=512
                )

                top_values, top_indices = T.topk(
                    class_dists, min(class_dists.size(0), nu_val)
                )  # get nu highest distances

                # fit weibull distributino on nu highest distances
                distribution = libmr.MR()
                distribution.fit_high(top_values, len(top_values))
                self.weibulls.append(distribution)
            else:
                print(f"WARNING:::: NO samples give for class {c}")
                self.weibulls.append(None)
                # raise ValueError(f'ERROR::: No samples give for class {c}')

    def forward_openmax(self, x: Tensor, alpha: Optional[int] = None) -> Tensor:
        x, latents, _ = self.forward(x)
        return self.update_probs(x, mavs=latents, alpha=alpha)


class CROSRLoss_(nn.Module):
    def forward(self, x, x_bar, z, y):
        return F.cross_entropy(z, y) + F.mse_loss(x_bar, x)


class CROSRLossWrapper(SupervisedLoss):
    def feed_model(self, model, x, y, mixed_precision):
        x, y, y_true = self.get_model_input(model, x, y, mixed_precision)
        return model(x)

    def forward(self, model, x, y, mixed_precision, training):
        z, latents, x_bar = self.feed_model(model, x, y, mixed_precision)

        if self.cache_labels_:
            self.cache_labels(z, y.to(z.device), training=model.training)

        return self.loss(x, x_bar, z, y), self.calc_metric(None, None)


def CROSRLoss(*args, **kwargs):
    return CROSRLossWrapper(CROSRLoss_)


# Better to use openauc or auroc for the instead of this function
@evaluation_function
def crosr_eval(
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

        # Calculate Recall for each unknown class individually
        for cls in unique_unknown_classes:
            cls_mask = y == cls
            # Check if any predictions correspond to this class and calculate accuracy
            cls_pred_correct = (y_pred[cls_mask] == -1).float().mean()
            zd_results[f"{label_prefix}zd_acc_class_{cls.item()}"] = (
                cls_pred_correct.item()
            )

        # for multiclass eval change -1 zd label to true labels
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
def crosr_engine(
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

    """
    x_train, latents_train, x_bar_train = model(x_train)
    x_test, latents_test, x_bar_test = model(x_test)
    """
    x_train, latents_train = model.get_latents(x_train, chunk_size=512)
    x_test, latents_test = model.get_latents(x_test, chunk_size=512)

    # -- fit openmax weibull distributions
    model.fit_weibull(
        x=x_train,
        y=y_train,
        latents=T.cat((x_train, latents_train), dim=-1),
        nu_val=nu,
    )

    osr_logits = model.update_probs(
        z=x_test,
        mavs=T.cat((x_test, latents_test), dim=-1),
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
