import torch as T
from torch import Tensor
import torch.nn as nn
from typing import List, Optional, Type
from model.model import make_mlp
from baseline_implementations.common.losses import BaseLoss
from baseline_implementations.common.model_eval import model_eval
import numpy as np
from util.metrics import mean_auroc


class AutoEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        neurons: List[int],
        dropout: float = 0.0,
        dropout_layer: Type[nn.Module] = nn.Dropout,
        final_layer_activation: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.mlp = make_mlp(
            d_in=d_in,
            neurons=neurons,
            d_out=d_in,
            dropout=dropout,
            dropout_layer=dropout_layer,
            final_layer_activation=final_layer_activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

    def forward_predict(self, x, threshold):
        x = self.forward(x)
        return self.predict(x, threshold)

    def predict(
        self,
        x,
        threshold: float,
    ):
        mse = self.calc_mse(x)
        y_pred = T.zeros(mse.size(0))
        y_pred[mse > threshold] = 1
        return y_pred

    def calc_mse(
        self,
        x,
        x_true,
    ):
        return T.mean((x - x_true) ** 2, dim=-1)


class SymetricAutoEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        neurons: List[int],
        bottleneck_ratio: float,
        dropout: float = 0.0,
        dropout_layer: Type[nn.Module] = nn.Dropout,
        final_layer_activation: Optional[Type[nn.Module]] = None,
    ) -> None:

        super().__init__()

        if not isinstance(neurons, list):
            neurons = [neurons]

        encoder_neurons = neurons
        bottleneck = int(bottleneck_ratio * d_in)
        decoder_neurons = neurons[::-1]

        self.encoder = make_mlp(
            d_in=d_in,
            neurons=encoder_neurons,
            d_out=bottleneck,
            dropout=dropout,
            dropout_layer=dropout_layer,
            final_layer_activation=nn.ReLU,
        )

        self.decoder = make_mlp(
            d_in=bottleneck,
            neurons=decoder_neurons,
            d_out=d_in,
            dropout=dropout,
            dropout_layer=dropout_layer,
            final_layer_activation=final_layer_activation,
        )

    def mlp(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    def encode(self, x, chunk_size=None):
        if chunk_size is None:
            return self.encoder(x)
        else:
            embeddings = []
            n_samples = x.size(0)
            chunk_i = 0

            for idx in range(0, n_samples, chunk_size):
                chunk_i += 1

                # chunk features to compare to all test samples
                x_chunk = x[
                    idx : min((idx + chunk_size), n_samples), :
                ]  # use rest of data if not enough for chunk
                z_chunk = self.encoder(x_chunk)
                embeddings.append(z_chunk.detach())

            embeddings = T.cat(embeddings, dim=0)

            return embeddings

    def decode(self, x):
        return self.decoder(x)

    def forward_predict(self, x, threshold):
        x = self.forward(x)
        return self.predict(x, threshold)

    def predict(
        self,
        x,
        threshold: float,
    ):
        mse = self.calc_mse(x)
        y_pred = T.zeros(mse.size(0))
        y_pred[mse > threshold] = 1
        return y_pred

    def get_mse_scores(self, x, chunk_size: Optional[int] = None):
        if chunk_size is None:
            return T.mean((x - self.forward(x)) ** 2, dim=-1)
        else:
            mses = []
            n_samples = x.size(0)
            chunk_i = 0

            for idx in range(0, n_samples, chunk_size):
                chunk_i += 1

                # chunk features to compare to all test samples
                x_chunk = x[
                    idx : min((idx + chunk_size), n_samples), :
                ]  # use rest of data if not enough for chunk
                mse = T.mean((x_chunk - self.forward(x_chunk)) ** 2, dim=-1)
                mses.append(mse.detach())

            return T.cat(mses, dim=0)

    def calc_mse(
        self,
        x,
        x_true,
    ):
        return T.mean((x - x_true) ** 2, dim=-1)


class MSELoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, loss=nn.MSELoss, **kwargs)

    def forward(self, model, x, y, mixed_precision, training):
        x, y, y_true = self.get_model_input(model, x, y, mixed_precision)
        xbar = model(x)
        return self.loss(xbar, x), self.calc_metric(x.clone().detach(), y_true)


def eval_autoencoder(
    x_test,
    z_test,
    y_test,
    x_train,
    z_train,
    y_train,
    threshold: float,
):
    y_test[y_test > 1] = 1
    y_train[y_train > 1] = 1

    mse = T.mean(
        (T.tensor(x_train, dtype=T.float32, device=z_train.device) - z_train) ** 2,
        dim=-1,
    )
    y_pred_train = T.zeros(mse.size(0))
    y_pred_train[mse > threshold] = 1
    train_results = model_eval(
        y_train,
        y_pred_train,
        label="train",
        return_class_level=True,
        return_detection_metrics=True,
    )

    mse = T.mean(
        (T.tensor(x_test, dtype=T.float32, device=z_test.device) - z_test) ** 2, dim=-1
    )
    y_pred_test = T.zeros(mse.size(0))
    y_pred_test[mse > threshold] = 1
    test_results = model_eval(
        y_test,
        y_pred_test,
        label="test",
        return_class_level=True,
        return_detection_metrics=True,
    )

    return {**train_results, **test_results}


def fit_autoencoder(
    x_val,
    z_val,
    y_val,
    threshold_vals=[
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ],
    label_prefix=None,
):
    y_val[y_val > 1] = 1
    results = []
    label = label_prefix if label_prefix is not None else ""
    x_val = T.tensor(x_val, dtype=T.float32, device=z_val.device)
    mse = T.mean((x_val - z_val) ** 2, dim=-1)

    for threshold in threshold_vals:
        y_pred = T.zeros(mse.size(0))
        y_pred[mse > threshold] = 1

        results_dict = model_eval(
            y_val,
            y_pred,
            label=label,
            return_class_level=True,
            return_detection_metrics=True,
        )
        results_dict["threshold"] = threshold
        results.append(results_dict)

    return results


def all_metric_eval(
    mse,
    y_true,
    threshold,
    label=None,
    return_class_level=True,
    return_detection_metrics=True,
    **kwargs,
):
    y_pred = T.zeros(mse.size(0))
    y_pred[mse > threshold] = 1
    return model_eval(
        y_true,
        y_pred,
        label=label,
        return_class_level=return_class_level,
        return_detection_metrics=return_detection_metrics,
    )


def fit_autoencoder_(
    x_val,
    z_val,
    y_val,
    eval_fn=all_metric_eval,
    threshold_vals=[
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ],
    label_prefix=None,
):
    y_val[y_val > 1] = 1
    results = []
    label = label_prefix if label_prefix is not None else ""
    x_val = T.tensor(x_val, dtype=T.float32, device=z_val.device)
    mse = T.mean((x_val - z_val) ** 2, dim=-1)

    for threshold in threshold_vals:
        results_dict = eval_fn(
            mse,
            y_val,
            threshold=threshold,
            label=label,
        )
        results_dict["threshold"] = threshold
        results.append(results_dict)

    return results


def auroc_eval(
    x,
    z,
    y,
    label_prefix=None,
    return_class_level: bool = False,
):
    """fit autoencoder based on macro mean auroc"""

    label_prefix = label_prefix or ""
    if not isinstance(x, Tensor):
        x = T.tensor(x, dtype=T.float32, device=z.device)

    mse = T.mean((x - z) ** 2, dim=-1)
    if return_class_level:
        class_auroc = mean_auroc(scores=mse, y_true=y, return_class_level=True)
        results = {
            f"{label_prefix}class_{i + 1}_auroc": score
            for i, score in enumerate(class_auroc)
        }
        results[f"{label_prefix}mean_auroc"] = np.mean(class_auroc)
    else:
        results = {
            f"{label_prefix}mean_auroc": mean_auroc(
                scores=mse, y_true=y, return_class_level=False
            )
        }

    return results
