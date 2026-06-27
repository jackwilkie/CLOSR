import torch as T
import torch.nn as nn
from torch import Tensor
from typing import Optional, List
import numpy as np
from numpy import ndarray
from baseline_implementations.common.model_eval import evaluation_function
from util.metrics import balanced_auroc
from baseline_implementations.common.process_batch import process_batch
from sklearn.mixture import GaussianMixture
from baseline_implementations.common.training_loops import train
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _make_dataloader(x, y, device, config):
    dataset_config = config.get("dataset", {}) if config is not None else {}
    batch_size = dataset_config.get("batch_size", 256)
    shuffle = dataset_config.get("shuffle", True)
    drop_last = dataset_config.get("drop_last", False)

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


def cosine_distance(x1: Tensor, x2: Tensor) -> Tensor:
    x1_norm = F.normalize(x1, dim=1)
    x2_norm = F.normalize(x2, dim=1)
    return 1 - T.sum(x1_norm * x2_norm, dim=1)


# -- engine
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
            scores = cosine_distance(x_data, z).cpu().detach().numpy()
        else:
            num_samples = x_data.shape[0]
            scores = []
            chunk_i = 0
            for idx in range(0, num_samples, chunk_size):
                chunk_i += 1
                x_chunk = x_data[idx : min((idx + chunk_size), num_samples), :]
                z_chunk = model(x_chunk)
                chunk_scores = cosine_distance(x_chunk, z_chunk).cpu().detach().numpy()
                scores.append(chunk_scores)
            scores = np.concatenate(scores, axis=0)

    elif dataloader is not None:
        scores = []
        for batch in dataloader:
            x, _ = process_batch(
                batch, device=device, mixed_precision=False, non_blocking=True
            )
            z = model(x)
            score = cosine_distance(x, z).cpu().detach().numpy()
            scores.append(score)
        scores = np.concatenate(scores, axis=0)
    else:
        raise ValueError("Exactly one of x_data and dataloader must be provided.")
    return scores


@T.no_grad()
@evaluation_function
def duad_engine(
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


# --- custom training function
def filter_data(
    X,
    n_components=10,
    top_k=2,
    random_state=None,
):
    """
    Select indices of data points belonging to the top_k clusters with the lowest variance.

    Parameters:
    - X: torch.Tensor of shape (n_samples, n_features)
        The input data.
    - n_components: int
        Number of GMM components.
    - top_k: int
        Number of clusters to select based on lowest variance.
    - random_state: int
        Random seed for reproducibility.

    Returns:
    - selected_indices: torch.Tensor
        Indices of data points in the selected clusters.
    """
    # Convert to NumPy for GMM

    X_np = X.cpu().numpy() if not isinstance(X, ndarray) else X

    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=random_state
    )
    gmm.fit(X_np)
    labels = gmm.predict(X_np)

    # Compute variance for each cluster
    variances = []
    for i in range(n_components):
        cluster_data = X_np[labels == i]
        if cluster_data.size == 0:
            variances.append((i, float("inf")))
            continue
        variance = np.var(cluster_data)
        variances.append((i, variance))

    # Sort clusters by variance in ascending order and select top_k clusters
    variances.sort(key=lambda x: x[1])
    selected_clusters = [cluster_id for cluster_id, _ in variances[:top_k]]

    # Get indices of data points in the selected clusters
    selected_indices = T.where(T.tensor(np.isin(labels, selected_clusters)))[0]

    return selected_indices


@evaluation_function
def get_encodings(
    model: nn.Module,
    x_data=None,
    y_data=None,
    dl=None,
    mixed_precision: bool = False,
    non_blocking: bool = True,
    batch: bool = True,
    chunk_size: Optional[int] = 1024,
    move_to_cpu: bool = True,
) -> Tensor:

    x_data = T.tensor(x_data, dtype=T.float32)
    y_data = T.tensor(y_data, dtype=T.int64)
    if dl is not None:
        # use data provided in dataloader if provided
        if batch:
            features, labels = [], []
            for batch in dl:
                x, y = process_batch(
                    batch,
                    device=next(model.parameters()).device,
                    mixed_precision=mixed_precision,
                    non_blocking=non_blocking,
                )
                z = model.encode(x)
                features.append(z.cpu().detach())
                labels.append(y.cpu().detach())

            features = T.cat(features, dim=0)
            labels = T.cat(labels, dim=0)

        else:
            # use data extracted from dataset, kept for legacy reasons
            x, labels = dl.dataset.x_data, dl.dataset.y_data
            features = model(x)

    elif x_data is not None:
        if chunk_size is None:
            # use provided data
            features = model(x_data).cpu().detach()
            labels = y_data.cpu().detach()
        else:
            features, labels = [], []

            n_samples = x_data.shape[0]
            chunk_i = 0

            for idx in range(0, n_samples, chunk_size):
                chunk_i += 1

                # chunk features to compare to all test samples
                x_chunk = x_data[
                    idx : min((idx + chunk_size), n_samples), :
                ]  # use rest of test data if not enough for chun
                y_chunk = y_data[idx : min((idx + chunk_size), n_samples)]
                x, y = process_batch(
                    (x_chunk, y_chunk),
                    device=next(model.parameters()).device,
                    mixed_precision=mixed_precision,
                    non_blocking=non_blocking,
                )
                z = model(x)

                z = z.cpu() if move_to_cpu else z
                y = y.cpu() if move_to_cpu else y

                features.append(z.detach())
                labels.append(y.detach())

            features = T.cat(features, dim=0)
            labels = T.cat(labels, dim=0)

    else:
        # raise error, either data or dl needed and both not provided
        raise ValueError("ERROR::: Need either data loader of x data")

    return features.detach().numpy()


def train_duad(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    device: str,
    config: dict,
    top_k: int = None,
    n_gmm: int = None,
    filter_every: int = 50,
    chunk_size: int = 512,
    train_dl=None,
    optimiser=None,
    criterion=None,
    schedule=None,
    scalar=None,
    logger=None,
):
    top_k = top_k or model.top_k
    n_gmm = n_gmm or model.n_gmm

    # -- init training
    if train_dl is None:
        train_dl = _make_dataloader(x_train, y_train, device, config)

    if optimiser is None or criterion is None:
        raise ValueError(
            "train_duad requires optimiser and criterion arguments. "
            "Create them in the calling training script."
        )

    # train model
    total_epochs = config["hyperparameters"]["training_epochs"]
    steps = (
        total_epochs // filter_every
        if total_epochs % filter_every == 0
        else (total_epochs // filter_every) + 1
    )
    epochs_per_step = total_epochs // steps
    remaining_epochs = total_epochs

    x_filtered = None

    while remaining_epochs > 0:
        # -- filter data
        if x_filtered is None:
            i_filtered = filter_data(
                x_train,
                n_components=n_gmm,
                top_k=top_k,
            )

            x_filtered = x_train[i_filtered]
            y_filtered = y_train[i_filtered]

            train_dl = _make_dataloader(x_filtered, y_filtered, device, config)

        else:
            # filter based on embedded space
            with T.no_grad():
                z_train = get_encodings(
                    model=model,
                    x_data=x_train,
                    y_data=y_train,
                    batch=True,
                    chunk_size=chunk_size,
                    move_to_cpu=True,
                )

            i_filtered = filter_data(
                z_train,
                n_components=n_gmm,
                top_k=top_k,
            )

            x_filtered = x_train[i_filtered]
            y_filtered = y_train[i_filtered]

            train_dl = _make_dataloader(x_filtered, y_filtered, device, config)

        # train
        _ = train(
            model=model,
            optimiser=optimiser,
            loss_calc=criterion,
            epochs=min(epochs_per_step, remaining_epochs),
            train_dl=train_dl,
            val_dl=None,
            scheduler=schedule,
            logger=logger,
            eval_func=None,
            eval_interval=1,
            ep_log_interval=1,
            scalar=scalar,
            grad_accumulation=config["hyperparameters"]["grad_accumulation"],
            checkpoint_interval=0,
            val_pass=False,
            grad_clipping=config["hyperparameters"].get("grad_clipping", None),
        )
        remaining_epochs -= epochs_per_step

    return None, None
