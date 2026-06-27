"""
Implementation the open auc evaluation metric as detailed in:

    https://arxiv.org/pdf/2210.13458

Created on: 08/08/24
"""

import numpy as np
from numpy import ndarray
from typing import Optional
from sklearn.metrics import (
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)


def calc_auc(
    scores: ndarray,
    labels: ndarray,
    base_classes,
    target_classes,
    balanced: bool = False,
):
    labels = labels.copy()
    labels[np.isin(labels, base_classes)] = -2

    if balanced:
        scores = scores[(labels == -2) | (np.isin(labels, target_classes))]
        labels = labels[(labels == -2) | (np.isin(labels, target_classes))]

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]

        roc_scores = []

        for c in unique_labels:
            _labels = labels.copy()
            _labels[_labels == c] = -1

            _scores = scores[_labels < 0]
            _labels = _labels[_labels < 0]
            roc_scores.append(roc_auc_score(_labels + 2, _scores))

        auc_score = np.mean(roc_scores)

    else:
        labels[np.isin(labels, target_classes)] = -1
        scores = scores[labels < 0]
        labels = labels[labels < 0]

        auc_score = roc_auc_score(labels + 2, scores)

    return auc_score


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95):
    y_true = np.asarray(y_true) == 1
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    recall = np.r_[recall[last_ind::-1], 1]
    fps = np.r_[fps[last_ind::-1], 0]

    cutoff = np.argmin(np.abs(recall - recall_level))
    return fps[cutoff] / np.sum(~y_true)


def balanced_fpr_at_recall(
    y_true,
    scores,
    recall_level: float = 0.95,
):
    fpr_scores = []

    for c in np.unique(y_true):
        if c == 0:
            # no auroc for benign data
            continue

        # get attack class and benign traffic
        class_mask = (y_true == 0) | (y_true == c)
        y = y_true[class_mask]
        y[y > 0] = 1
        x = scores[class_mask]
        fpr_scores.append(fpr_and_fdr_at_recall(y, x, recall_level=recall_level))

    return fpr_scores


def open_auc(
    probs: ndarray,
    osr_scores: ndarray,
    y_true: ndarray,
    balanced: bool = False,
    balanced_osr: Optional[bool] = None,
) -> float:

    balanced_osr = balanced_osr or balanced
    n_closed_set = probs.shape[-1]
    n_open_set = len(np.unique(y_true)) - n_closed_set

    # -- calculate auroc on openset
    osr_labels = y_true.copy()
    osr_labels[osr_labels < n_closed_set] = 0  # cloed set labels set to 0
    osr_labels[osr_labels > 0] = osr_labels[osr_labels > 0] - (
        n_closed_set - 1
    )  # open set labels increasing from 1

    if balanced_osr:
        auroc = np.mean(
            [
                roc_auc_score(
                    osr_labels[(osr_labels == 0) | (osr_labels == c + 1)],
                    osr_scores[(osr_labels == 0) | (osr_labels == c + 1)],
                )
                for c in range(n_open_set)
            ]
        )

    else:
        osr_labels[osr_labels > 1] = 1
        auroc = roc_auc_score(
            osr_labels, osr_scores
        )  # osr scores with all open set classes pooled into one

    # -- calculate closed set performance
    closed_set_probs = probs[y_true < n_closed_set]
    closed_set_labels = y_true[y_true < n_closed_set]
    closed_set_preds = np.argmax(closed_set_probs, axis=1)
    acc = (
        recall_score(closed_set_labels, closed_set_preds, average="macro")
        if balanced
        else accuracy_score(closed_set_labels, closed_set_preds)
    )

    # print(auroc)
    # print(acc)

    # -- metric is product of closed set and open set performance
    return auroc * acc


def osr_eval(
    probs: ndarray,
    osr_scores: ndarray,
    y_true: ndarray,
    balanced: bool = False,
    balanced_osr: Optional[bool] = None,
) -> float:

    balanced_osr = balanced_osr or balanced
    n_closed_set = probs.shape[-1]
    n_open_set = len(np.unique(y_true)) - n_closed_set

    # -- calculate auroc on openset
    osr_labels = y_true.copy()
    osr_labels[osr_labels < n_closed_set] = 0  # cloed set labels set to 0
    osr_labels[osr_labels > 0] = osr_labels[osr_labels > 0] - (
        n_closed_set - 1
    )  # open set labels increasing from 1
    osr_labels_binary = osr_labels.copy()
    osr_labels_binary[osr_labels > 0] = 1

    if balanced_osr:
        auroc = np.mean(
            [
                roc_auc_score(
                    osr_labels[(osr_labels == 0) | (osr_labels == c + 1)],
                    osr_scores[(osr_labels == 0) | (osr_labels == c + 1)],
                )
                for c in range(n_open_set)
            ]
        )

    else:
        osr_labels[osr_labels > 1] = 1
        auroc = roc_auc_score(
            osr_labels, osr_scores
        )  # osr scores with all open set classes pooled into one

    # -- calculate closed set performance
    closed_set_probs = probs[y_true < n_closed_set]
    closed_set_labels = y_true[y_true < n_closed_set]
    closed_set_preds = np.argmax(closed_set_probs, axis=1)
    acc = (
        recall_score(closed_set_labels, closed_set_preds, average="macro")
        if balanced
        else accuracy_score(closed_set_labels, closed_set_preds)
    )

    # -- metric is product of closed set and open set performance
    metrics = dict(
        accuracy=acc,
        openset_auc=auroc,
        open_auc=acc * auroc,
        fpr_95=fpr_and_fdr_at_recall(osr_labels_binary, osr_scores),
        fpr_95_binary=fpr_and_fdr_at_recall(
            osr_labels_binary[(y_true == 0) | (osr_labels_binary == 1)],
            osr_scores[(y_true == 0) | (osr_labels_binary == 1)],
        ),
    )

    return metrics


# -- balanced AUC-PR calculation
def calc_auc_pr(
    y_true,
    scores,
):
    return average_precision_score(y_true, scores)


def balanced_auc_pr(
    y_true,
    scores,
):
    pr_scores = []

    for c in np.unique(y_true):
        if c == 0:
            # no auroc for benign data
            continue

        # get attack class and benign traffic
        class_mask = (y_true == 0) | (y_true == c)
        y = y_true[class_mask]
        y[y > 0] = 1
        x = scores[class_mask]
        pr_scores.append(calc_auc_pr(y, x))

    return pr_scores
