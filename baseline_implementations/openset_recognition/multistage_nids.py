"""
Implementation of the multistage NIDS detailed in:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10077796

Created on: 14/08/24
"""

import numpy as np
from numpy import ndarray

from sklearn.metrics import roc_auc_score
from typing import Optional


# ========== Evaluation Functions ==========


def evenly_spaced_samples(scores, n):
    # Sort the list of scores
    sorted_scores = sorted(scores)

    # Total number of scores
    total_scores = len(sorted_scores)

    # Calculate indices for evenly spaced samples
    if n >= total_scores:
        return (
            sorted_scores  # Return all scores if n is greater than the number of scores
        )
    else:
        indices = [int(i * total_scores / n) for i in range(n)]

    # Select samples from the sorted list
    sampled_scores = [sorted_scores[index] for index in indices]

    return sampled_scores


def fit_anomaly_thres(
    anomaly_scores: ndarray,
    cls_scores: ndarray,
    y_train: ndarray,
    return_balanced: bool = False,
    n_samples: int = 100,
):
    """Calculate best threshold for seperateing benign and known malicous traffic"""

    best_acc = 0.0
    best_thres = None

    best_recall = 0.0
    best_balanced_thres = None

    attack_pred = np.argmax(cls_scores, axis=-1) + 1
    mal_scores = np.max(cls_scores, axis=-1)

    for thres_b in evenly_spaced_samples(anomaly_scores, n_samples):
        for thres_m in evenly_spaced_samples(mal_scores, n_samples):
            # get predictions
            y_pred = attack_pred.copy()
            y_pred[(anomaly_scores <= thres_b) | (mal_scores <= thres_m)] = 0

            # get accuracy
            acc = np.sum(y_pred == y_train) / y_pred.shape[0]
            balanced_recall = np.mean(
                [
                    np.sum(y_pred[y_train == c] == c) / y_pred[y_train == c].shape[0]
                    for c in np.unique(y_train)
                ]
            )

            if acc > best_acc:
                best_acc = acc
                best_thres = (thres_b, thres_m)

            if balanced_recall > best_recall:
                best_recall = balanced_recall
                best_balanced_thres = (thres_b, thres_m)

    if return_balanced:
        return best_thres, best_balanced_thres
    else:
        return best_thres


def calc_auc(
    scores: ndarray,
    labels: ndarray,
    base_classes: Optional[ndarray] = None,
    target_classes: Optional[ndarray] = None,
    balanced: bool = False,
):
    labels = labels.copy()

    if base_classes is not None:
        labels[np.isin(labels, base_classes)] = 0

    if balanced:
        if target_classes is not None:
            labels = labels[(labels == 0) | (np.isins(labels, target_classes))]

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]

        auc_score = np.mean(
            [
                roc_auc_score(
                    labels[(labels == 0) | (labels == c)],
                    scores[(labels == 0) | (labels == c)],
                )
                for c in unique_labels
            ]
        )

    else:
        if target_classes is not None:
            labels[np.isin(labels, target_classes)] = 1
            scores = scores[labels <= 1]
            labels = labels[labels <= 1]

        auc_score = roc_auc_score(labels, scores)

    return auc_score


def make_predictions(
    ad_scores: ndarray,
    cls_scores: ndarray,
    y_test: ndarray,
    thres_b: float,
    thres_m: float,
    balanced: bool = False,
):

    n_classes = cls_scores.shape[-1] + 1
    y_pred = np.argmax(cls_scores, axis=-1)
    y_pred = y_pred + 1

    m_scores = np.max(cls_scores, axis=-1)
    y_pred[(ad_scores <= thres_b) | (m_scores <= thres_m)] = 0

    # closed set score is acc, or mean recall if balanced
    if balanced:
        closed_set_score = (y_pred == y_test) / y_pred.shape[0]
    else:
        closed_set_score = np.mean(
            [
                (y_pred[y_test == c] == y_test) / y_pred[y_test == c].shape[0]
                for c in np.unique(y_test)
            ]
        )

    """
    To calculate AUC:
    1) malicous traffic must be separated from benign and zd
    2) benign and zd must then be seperated from each other
    Thus AUC is product of both AUC metrics.
    Can be made balanced by meaning per class values
    """

    open_set_score = 0.0

    # calculate auc for separating benign traffic
    unique_labels = np.unique(y_test)
    mal_labels = unique_labels[unique_labels > 0]
    known_labels = mal_labels[mal_labels < n_classes]
    zd_labels = mal_labels[mal_labels >= n_classes]

    mal_auc = calc_auc(
        scores=-1 * m_scores,
        labels=y_test,
        base_classes=known_labels,
        target_classes=np.concatenate((np.array(0), zd_labels), axis=0),
        balanced=balanced,
    )

    # calculate auc for separating known attacks
    benign_zd_auc = calc_auc(
        scores=ad_scores,
        labels=y_test,
        base_classes=np.array(0),
        target_classes=zd_labels,
        balanced=balanced,
    )

    open_set_score = mal_auc * benign_zd_auc

    return closed_set_score * open_set_score


def calc_metric(
    ad_scores: ndarray,
    cls_scores: ndarray,
    y_test: ndarray,
    thres_b: float,
    thres_m: float,
    balanced: bool = False,
):
    """
    Calculate best closed set performance.
    Threshold is selected to maximise mean recall when balanced, otherwise accuracy
    """

    n_classes = y_test.shape[-1] + 1
    y_pred = np.argmax(cls_scores, axis=-1)
    y_pred = y_pred + 1

    m_scores = np.max(cls_scores, axis=-1)
    y_pred[(ad_scores <= thres_b) | (m_scores <= thres_m)] = 0

    # closed set score is acc, or mean recall if balanced
    if balanced:
        closed_set_score = (y_pred == y_test) / y_pred.shape[0]
    else:
        closed_set_score = np.mean(
            [
                (y_pred[y_test == c] == y_test) / y_pred[y_test == c].shape[0]
                for c in np.unique(y_test)
            ]
        )

    """
    To calculate AUC:
    1) malicous traffic must be separated from benign and zd
    2) benign and zd must then be seperated from each other
    Thus AUC is product of both AUC metrics.
    Can be made balanced by meaning per class values
    """

    open_set_score = 0.0

    # calculate auc for separating benign traffic
    unique_labels = np.unique(y_test)
    mal_labels = unique_labels[unique_labels > 0]
    known_labels = mal_labels[mal_labels < n_classes]
    zd_labels = mal_labels[mal_labels >= n_classes]

    mal_auc = calc_auc(
        scores=-1 * m_scores,
        labels=y_test,
        base_classes=known_labels,
        target_classes=np.concatenate((np.array(0), zd_labels), axis=0),
        balanced=balanced,
    )

    # calculate auc for separating known attacks
    benign_zd_auc = calc_auc(
        scores=ad_scores,
        labels=y_test,
        base_classes=np.array(0),
        target_classes=zd_labels,
        balanced=balanced,
    )

    open_set_score = mal_auc * benign_zd_auc

    return closed_set_score * open_set_score


# ========== Get Open Set Scores ==========
def score_relative_to_class(train_scores, test_scores):
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)

    # Sort train scores once
    sorted_train = np.sort(train_scores)

    # For each test score, count how many train scores are less than it
    # This uses binary search, so it's fast
    ranks = np.searchsorted(sorted_train, test_scores, side="left")

    # Optionally normalize to get a percentile [0, 1]
    percentile_scores = ranks / len(train_scores)
    return percentile_scores


def get_osr_scores(
    ad_scores_train,
    m_scores_train,
    ad_scores_test,
    m_scores_test,
    y_train,
):
    # probability of separating from benign traffic
    b_scores = score_relative_to_class(ad_scores_train[y_train == 0], ad_scores_test)
    print(b_scores)
    print(b_scores.shape)

    # probaility of separting from malicious traffic
    m_scores = score_relative_to_class(-m_scores_train[y_train > 0], -m_scores_test)
    print(m_scores)
    print(m_scores.shape)
    return b_scores * m_scores
