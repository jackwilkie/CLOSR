#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model evaluation functions

Created on Mon Jun 12 18:52:12 2023

@author: jack
"""

# import required modules
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import torch.nn as nn
import torch as T
from torch import Tensor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize


# ---------------- Torch eval context managers --------------------------
class EvalContext:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.is_torch = isinstance(model, nn.Module)
        self.is_training = model.training if self.is_torch else False

    def __enter__(self) -> None:
        if self.is_torch:
            self.model.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_training and self.is_torch:
            self.model.train()


class TrainContext:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.is_training = model.training

    def __enter__(self):
        self.model.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_training:
            self.model.eval()


def evaluation_function(func):
    def wrapper(*args, **kwargs):
        # model must be first arg
        if "model" in kwargs:
            model = kwargs["model"]
        else:
            model = args[0]

        # if not isinstance(model, nn.Module):
        #    raise ValueError(f'model must be a nn.Module, got {model}')

        with T.no_grad():
            with EvalContext(model):
                # return func (*args, **kwargs)
                return func(*args, **kwargs)

    return wrapper


def train_function(func):
    def wrapper(*args, **kwargs):
        # model must be first arg
        if "model" in kwargs:
            model = kwargs["model"]
        else:
            model = args[0]

        if not isinstance(model, nn.Module):
            raise ValueError(f"model must be a nn.Module, got {model}")

        with TrainContext(model):
            # return func (*args, **kwargs)
            return func(*args, **kwargs)

    return wrapper


# ---------------- Model Metric Calculation Functions --------------------------


def model_eval(
    y_true,
    y_pred,
    label="",
    return_class_level: bool = False,
    return_detection_metrics: bool = False,
):
    """
    Function to get model performance metrics from ground truth and predicted labels.

    Parameters
    --------
    y_true : Numpy Array
        Ground truth class labels.
    y_pred : Numpy Array
        Predicted class labels.
    label : Str
         Used to name results in dictionary.

    Returns
    -------
    results_dict : Dictionary
        Dictionary of model performance metrics.

    """
    label = label if label is not None else ""
    if len(label) > 0:
        label += "_"
    results_dict = {}  # initalise dictionary to store model performance

    cm = confusion_matrix(y_true, y_pred)

    acc = np.trace(cm) / np.sum(cm)  # find model accuracy
    results_dict[f"{label}Acc"] = acc

    # find precision and recall scores
    precision_scores = [
        (cm[i][i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0.0 else 0.0)
        for i in range(len(cm))
    ]  # get precision for each class
    recall_scores = [
        (cm[i][i] / np.sum(cm[i]) if np.sum(cm[i]) > 0 else 0.0) for i in range(len(cm))
    ]  # calculate recall value for each class

    if return_class_level:
        # store precision and recall values in dictionary
        for i, recall in enumerate(recall_scores):
            results_dict[f"{label}Class_" + str(i) + "_Recall"] = recall

        for i, precision in enumerate(precision_scores):
            results_dict[f"{label}Class_" + str(i) + "_Precision"] = precision

    # get mean recall value
    results_dict[f"{label}Recall"] = np.mean(recall_scores)

    # get mean precision
    results_dict[f"{label}Precision"] = np.mean(precision_scores)

    # calculate F1 score using different averaging
    results_dict[f"{label}Macro_F1"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0.0
    )
    results_dict[f"{label}Micro_F1"] = f1_score(
        y_true, y_pred, average="micro", zero_division=0.0
    )
    results_dict[f"{label}Weighted_F1"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0.0
    )

    if return_detection_metrics:
        benign_recall = recall_scores[0]
        results_dict[f"{label}fp_rate"] = 1 - benign_recall
        results_dict[f"{label}detection_rate"] = np.mean(recall_scores[1:])

    return results_dict


def get_metric_dict(y_true, y_pred):
    """
    Function to get model performance metrics from ground truth and predicted labels.

    Parameters
    --------
    y_true : Numpy Array
        Ground truth class labels.
    y_pred : Numpy Array
        Predicted class labels.
    """

    results_dict = {}  # initalise dictionary to store model performance

    cm = confusion_matrix(y_true, y_pred)

    results_dict["Accuracy"] = np.trace(cm) / np.sum(cm)  # find model accuracy

    # find precision and recall scores
    precision_scores = [
        (cm[i][i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0.0 else 0.0)
        for i in range(len(cm))
    ]  # get precision for each class
    recall_scores = [
        (cm[i][i] / np.sum(cm[i]) if np.sum(cm[i]) > 0 else 0.0) for i in range(len(cm))
    ]  # calculate recall value for each class

    # store precision and recall values in dictionary
    for i, recall in enumerate(recall_scores):
        results_dict[f"Class_{i}_Recall"] = recall

    for i, precision in enumerate(precision_scores):
        results_dict[f"Class_{i}_Precision"] = precision

    # get mean recall value
    results_dict["Recall"] = np.mean(recall_scores)

    # get mean precision
    results_dict["Precision"] = np.mean(precision_scores)

    # calculate F1 score using different averaging
    results_dict["Macro_F1"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0.0
    )
    results_dict["Micro_F1"] = f1_score(
        y_true, y_pred, average="micro", zero_division=0.0
    )
    results_dict["Weighted_F1"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0.0
    )

    results_dict["fp_rate"] = 1 - results_dict["Class_0_Recall"]

    class_dr = np.array(
        [
            np.count_nonzero(y_pred[y_true == class_num] == class_num)
            / np.count_nonzero(y_true == class_num)
            for class_num in np.unique(y_true)
            if class_num != 0
        ]
    )

    # results_dict['detection_rate'] = statistics.harmonic_mean(class_dr)
    results_dict["detection_rate"] = np.mean(class_dr)
    return results_dict


def mean_auroc(
    scores,
    y_true,
    return_class_level: bool = False,
    include_lower_thres: bool = True,
):
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().detach().numpy()

    if isinstance(scores, Tensor):
        scores = scores.cpu().detach().numpy()

    roc_scores = []

    for c in np.unique(y_true):
        if c == 0:
            # no auroc for benign data
            continue

        # get attack class and benign traffic
        class_mask = (y_true == 0) | (y_true == c)
        y = y_true[class_mask]
        y[y > 0] = 1

        x = scores[class_mask]
        roc = roc_auc_score(y, x)
        if include_lower_thres:
            roc_scores.append(max(roc, 1 - roc))
        else:
            roc_scores.append(roc)

    roc_scores.append(np.mean(roc_scores))

    if return_class_level:
        return roc_scores
    else:
        return roc_scores[-1]


def balanced_auroc(
    scores, labels, return_class_level: bool = False, include_lower_thres=False
):
    class_auroc = mean_auroc(
        scores=scores,
        y_true=labels,
        return_class_level=True,
        include_lower_thres=include_lower_thres,
    )
    class_auroc = (
        [max(x, 1 - x) for x in class_auroc[:-1]]
        if include_lower_thres
        else class_auroc
    )

    if return_class_level:
        return class_auroc
    else:
        return np.mean(class_auroc)


def macro_pr_auc(
    y_score,
    y_true,
):
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)  # shape: (B, C)
    return average_precision_score(y_true_bin, y_score, average="macro")
