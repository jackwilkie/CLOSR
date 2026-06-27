import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor
from typing import List, Optional, Tuple, Union


class EvalContext:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.is_torch = isinstance(model, nn.Module)
        self.was_training = model.training if self.is_torch else False

    def __enter__(self):
        if self.is_torch:
            self.model.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_torch and self.was_training:
            self.model.train()


def evaluation_function(func):
    def wrapper(*args, **kwargs):
        model = kwargs["model"] if "model" in kwargs else args[0]
        with T.no_grad():
            with EvalContext(model):
                return func(*args, **kwargs)

    return wrapper


def _as_tensor(x, dtype, device) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return T.tensor(x, dtype=dtype, device=device)


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.cpu().detach().numpy()
    return np.asarray(x)


def _get_device(model, device=None):
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _append_ood(x_test: Tensor, y_test, x_zd: Optional[Tensor] = None, y_zd=None):
    if x_zd is None:
        return x_test, y_test

    x_test = T.cat((x_test, x_zd), dim=0)
    y_test = np.concatenate((_as_numpy(y_test), _as_numpy(y_zd)), axis=0)
    return x_test, y_test


def cosine_similarity(x1: Tensor, x2: Tensor) -> Tensor:
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return x1 @ x2.t()


def distance_matrix(x1: Tensor, x2: Tensor, metric: str = "cosine") -> Tensor:
    if metric == "euclidean":
        return T.cdist(x1, x2, p=2.0)
    if metric == "manhattan":
        return T.cdist(x1, x2, p=1.0)
    if metric == "cosine":
        return 1 - cosine_similarity(x1, x2)
    raise ValueError(f"Invalid distance metric: {metric}")


def similarity_matrix(x1: Tensor, x2: Tensor, metric: str = "cosine") -> Tensor:
    if metric == "cosine":
        return cosine_similarity(x1, x2)
    return -distance_matrix(x1, x2, metric)


def apply_knn_weight(
    scores: Tensor, weight_fn: str = "hard", temp: Optional[float] = None
) -> Tensor:
    if weight_fn == "hard":
        return T.ones_like(scores)
    if weight_fn == "soft":
        return scores
    if weight_fn == "dino":
        if temp is None:
            raise ValueError('temp must be provided when weight_fn="dino"')
        return scores.div(temp).exp()
    raise ValueError(f"Invalid KNN weight function: {weight_fn}")


class OnlineContrastiveLoss(nn.Module):
    def __init__(
        self,
        m: Optional[float] = 1.0,
        distance_metric: str = "cosine",
        use_negative: bool = True,
        squared: bool = False,
        eps: float = 1e-16,
    ) -> None:
        super().__init__()
        self.m = m
        self.distance_metric = distance_metric
        self.use_negative = use_negative
        self.squared = squared
        self.eps = eps
        self.frac_pos = 0.0

    def get_fraction_pos(self):
        return self.frac_pos

    def forward(self, x1: Tensor, y: Tensor, x2: Optional[Tensor] = None) -> Tensor:
        y = y.clone()
        y[y > 0] = 1
        x2 = x1 if x2 is None else x2
        dists = distance_matrix(x1, x2, self.distance_metric)

        sim_labels = T.eq(y.unsqueeze(0), y.unsqueeze(1))
        sim_dists = dists * sim_labels
        if self.squared:
            sim_dists = T.pow(sim_dists, 2)

        num_valid_sim = T.sum(T.greater(sim_labels, self.eps).float())
        loss = T.sum(sim_dists) / (num_valid_sim + self.eps)
        self.frac_pos = 0.0

        if self.use_negative:
            if self.m is not None:
                dissim_dists = F.relu(self.m - dists)
            else:
                dissim_dists = -dists

            if self.squared:
                dissim_dists = T.pow(dissim_dists, 2)

            dissim_dists = dissim_dists * (~sim_labels)
            n_dissim = T.sum((~sim_labels).float())
            n_valid_dissim = T.sum(T.greater(dissim_dists, self.eps).float())
            self.frac_pos = n_valid_dissim / (n_dissim + self.eps)
            loss = loss + (T.sum(dissim_dists) / (n_valid_dissim + self.eps))

        return loss


Loss = OnlineContrastiveLoss


def model_eval(
    y_true,
    y_pred,
    label: str = "",
    return_class_level: bool = True,
    return_detection_metrics: bool = True,
) -> dict:
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    label = f"{label}_" if label else ""
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    results = {
        f"{label}Acc": np.trace(cm) / np.sum(cm),
        f"{label}Macro_F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0.0
        ),
        f"{label}Micro_F1": f1_score(
            y_true, y_pred, average="micro", zero_division=0.0
        ),
        f"{label}Weighted_F1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0.0
        ),
    }

    recalls = []
    precisions = []
    for i, class_label in enumerate(labels):
        precision = cm[i][i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0.0 else 0.0
        recall = cm[i][i] / np.sum(cm[i]) if np.sum(cm[i]) > 0.0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

        if return_class_level:
            results[f"{label}Class_{class_label}_Recall"] = recall
            results[f"{label}Class_{class_label}_Precision"] = precision

    results[f"{label}Recall"] = np.mean(recalls)
    results[f"{label}Precision"] = np.mean(precisions)

    if return_detection_metrics:
        benign_idx = np.where(labels == 0)[0]
        benign_recall = recalls[benign_idx[0]] if benign_idx.size > 0 else 0.0
        attack_recalls = [
            recall for class_label, recall in zip(labels, recalls) if class_label != 0
        ]
        results[f"{label}fp_rate"] = 1 - benign_recall
        results[f"{label}detection_rate"] = (
            np.mean(attack_recalls) if attack_recalls else 0.0
        )

    return results


def fpr_at_recall(y_true, y_score, recall_level: float = 0.95) -> float:
    y_true = np.asarray(y_true) == 1
    if np.sum(y_true) == 0 or np.sum(~y_true) == 0:
        return np.nan

    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    recall = tps / tps[-1]
    cutoff = np.argmin(np.abs(recall - recall_level))
    return fps[cutoff] / np.sum(~y_true)


def safe_roc_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def open_set_eval(
    probs: np.ndarray,
    osr_scores: np.ndarray,
    y_true,
    balanced: bool = False,
    balanced_osr: Optional[bool] = None,
) -> dict:
    y_true = _as_numpy(y_true).copy()
    balanced_osr = balanced if balanced_osr is None else balanced_osr
    n_closed_set = probs.shape[-1]
    n_open_set = len(np.unique(y_true)) - n_closed_set

    osr_labels = y_true.copy()
    osr_labels[osr_labels < n_closed_set] = 0
    osr_labels[osr_labels > 0] = osr_labels[osr_labels > 0] - (n_closed_set - 1)
    osr_labels_binary = osr_labels.copy()
    osr_labels_binary[osr_labels_binary > 0] = 1

    if balanced_osr and n_open_set > 0:
        aucs = []
        for c in range(n_open_set):
            mask = (osr_labels == 0) | (osr_labels == c + 1)
            aucs.append(safe_roc_auc(osr_labels[mask], osr_scores[mask]))
        openset_auc = np.nanmean(aucs)
    else:
        openset_auc = safe_roc_auc(osr_labels_binary, osr_scores)

    closed_mask = y_true < n_closed_set
    closed_set_probs = probs[closed_mask]
    closed_set_labels = y_true[closed_mask]
    closed_set_preds = np.argmax(closed_set_probs, axis=1)
    accuracy = (
        recall_score(
            closed_set_labels, closed_set_preds, average="macro", zero_division=0.0
        )
        if balanced
        else accuracy_score(closed_set_labels, closed_set_preds)
    )

    return {
        "accuracy": accuracy,
        "openset_auc": openset_auc,
        "open_auc": accuracy * openset_auc,
        "fpr_95": fpr_at_recall(osr_labels_binary, osr_scores),
        "fpr_95_binary": fpr_at_recall(
            osr_labels_binary[(y_true == 0) | (osr_labels_binary == 1)],
            osr_scores[(y_true == 0) | (osr_labels_binary == 1)],
        ),
    }


@T.no_grad()
def knn_predict(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    k: int = 5,
    dist_fn: str = "cosine",
    weight_fn: str = "hard",
    num_chunks: int = 100,
    num_classes: Optional[int] = None,
    temp: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(k, int):
        k = int(k)

    class_values = T.unique(y_train)
    if num_classes is None:
        num_classes = class_values.size(0)

    n_test = x_test.size(0)
    samples_per_chunk = max(n_test // num_chunks, 1)
    score_list = []
    pred_list = []

    for idx in range(0, n_test, samples_per_chunk):
        x_chunk = x_test[idx : min(idx + samples_per_chunk, n_test)]
        sims = similarity_matrix(x_chunk, x_train, dist_fn)
        k_ = min(k, sims.size(1))
        neighbour_scores, neighbour_indices = sims.topk(k_, largest=True, sorted=True)
        neighbour_labels = (
            y_train.view(1, -1).expand(x_chunk.size(0), -1).gather(1, neighbour_indices)
        )
        weights = apply_knn_weight(neighbour_scores, weight_fn=weight_fn, temp=temp)

        class_scores = T.zeros(
            x_chunk.size(0), num_classes, dtype=weights.dtype, device=x_train.device
        )
        for i, cls in enumerate(class_values):
            class_scores[:, i] = (weights * (neighbour_labels == cls)).sum(dim=1)

        pred_indices = T.argmax(class_scores, dim=1)
        pred_labels = class_values[pred_indices]
        score_list.append(class_scores.cpu().detach().numpy())
        pred_list.append(pred_labels.cpu().detach().numpy())

    return np.concatenate(score_list, axis=0), np.concatenate(pred_list, axis=0)


@T.no_grad()
def class_knn_distances(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    k: int,
    dist_fn: str = "cosine",
    chunk_size: Optional[int] = None,
) -> Tensor:
    classes = T.unique(y_train)
    scores = []
    n_test = x_test.size(0)
    samples_per_chunk = chunk_size or max(n_test // 100, 1)

    for cls in classes:
        class_train = x_train[y_train == cls]
        class_scores = []

        for idx in range(0, n_test, samples_per_chunk):
            x_chunk = x_test[idx : min(idx + samples_per_chunk, n_test)]
            dists = distance_matrix(x_chunk, class_train, dist_fn)
            k_ = min(k, dists.size(1))
            class_scores.append(
                dists.topk(k_, largest=False, sorted=True).values.mean(dim=-1)
            )

        scores.append(T.cat(class_scores, dim=0))

    return T.stack(scores, dim=-1)


@evaluation_function
def knn_eval(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    k: int = 5,
    dist_fn: str = "cosine",
    weight_fn: str = "hard",
    num_chunks: int = 100,
    num_classes: Optional[int] = None,
    temp: Optional[float] = None,
    label_prefix: Optional[str] = None,
    device=None,
) -> dict:
    device = _get_device(model, device)
    x_train = _as_tensor(x_train, T.float32, device)
    y_train = _as_tensor(y_train, T.int64, device)
    x_test = _as_tensor(x_test, T.float32, device)

    z_train = model(x_train)
    z_test = model(x_test)
    _, y_pred = knn_predict(
        x_train=z_train,
        y_train=y_train,
        x_test=z_test,
        k=k,
        dist_fn=dist_fn,
        weight_fn=weight_fn,
        num_chunks=num_chunks,
        num_classes=num_classes,
        temp=temp,
    )

    return model_eval(_as_numpy(y_test), y_pred, label=label_prefix or "")


@evaluation_function
def knn_min_distance_ood_eval(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    k: Union[int, List[int]] = 5,
    dist_fn: str = "cosine",
    chunk_size: Optional[int] = None,
    x_zd: Optional[Tensor] = None,
    y_zd=None,
    device=None,
) -> Union[dict, List[dict]]:
    device = _get_device(model, device)
    k_vals = k if isinstance(k, list) else [k]

    x_train = _as_tensor(x_train, T.float32, device)
    y_train = _as_tensor(y_train, T.int64, device)
    x_test = _as_tensor(x_test, T.float32, device)
    if x_zd is not None:
        x_zd = _as_tensor(x_zd, T.float32, device)
    x_test, y_test = _append_ood(x_test, y_test, x_zd, y_zd)
    y_test = _as_numpy(y_test)

    z_train = model(x_train)
    z_test = model(x_test)

    results = []
    for k_val in k_vals:
        class_dists = class_knn_distances(
            x_train=z_train,
            y_train=y_train,
            x_test=z_test,
            k=int(k_val),
            dist_fn=dist_fn,
            chunk_size=chunk_size,
        )
        class_scores = -class_dists
        osr_scores = T.min(class_dists, dim=-1).values
        probs = class_scores.cpu().detach().numpy()
        ood_scores = osr_scores.cpu().detach().numpy()

        metrics = open_set_eval(
            probs=probs, osr_scores=ood_scores, y_true=y_test, balanced=False
        )
        balanced_metrics = open_set_eval(
            probs=probs, osr_scores=ood_scores, y_true=y_test, balanced=True
        )
        results.append(
            {
                "k": int(k_val),
                "accuracy": metrics["accuracy"],
                "openset_auc": metrics["openset_auc"],
                "OpenAUC": metrics["open_auc"],
                "fpr_95": metrics["fpr_95"],
                "fpr_95_binary": metrics["fpr_95_binary"],
                "balanced_accuracy": balanced_metrics["accuracy"],
                "balanced_openset_auc": balanced_metrics["openset_auc"],
                "balanced_OpenAUC": balanced_metrics["open_auc"],
            }
        )

    return results if isinstance(k, list) else results[0]
