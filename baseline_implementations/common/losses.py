#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class for Loss Calculations

Created on Thu Sep 21 16:46:05 2023

@author: jack
"""

import torch as T
from torch import Tensor
import torch.nn as nn
from baseline_implementations.common.process_batch import process_batch
from .model_eval import model_eval
from baseline_implementations.common.training_loops import gather_concat


def identity(x):
    return x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args):

        if len(args) == 1:
            return args[0]
        else:
            return args


class BaseLoss(nn.Module):
    def __init__(self, loss, *args, batch_aug=Identity(), **kwargs):
        super().__init__()
        self.loss = loss(*args, **kwargs)
        self.batch_aug = batch_aug

    def forward(self, model, x, y, mixed_precision, training):
        logits, y_pred, y_true = self.feed_model(model, x, y, mixed_precision)
        return self.loss(logits, y_pred), self.calc_metric(
            logits.clone().detach(), y_true
        )

    def get_batch(self, model: nn.Module, x: Tensor, y: Tensor, mixed_precision):
        device = next(model.parameters()).device
        x, y = process_batch((x, y), device, mixed_precision)
        return x, y

    def apply_aug(self, model: nn.Module, x: Tensor, y: Tensor):
        y_true = y.clone()
        if model.training:
            x, y = self.batch_aug(x, y)
        return x, y, y_true

    def get_model_input(self, model: nn.Module, x: Tensor, y: Tensor, mixed_precision):
        x, y = self.get_batch(model, x, y, mixed_precision)
        x, y, y_true = self.apply_aug(model, x, y)
        return x, y, y_true

    def feed_model(self, model, x, y, mixed_precision):
        x, y, y_true = self.get_model_input(model, x, y, mixed_precision)
        x = model(x)
        return x, y, y_true

    def process_batch(self, x, y):
        return self.batch_aug(x, y)

    def calc_metric(self, logits, y_true):
        return T.tensor(0.0)

    def get_epoch_metrics(self, training=True, world_size=0):
        return {}


class SupervisedLoss(BaseLoss):
    def __init__(
        self, loss, *args, batch_aug=Identity(), cache_labels: bool = True, **kwargs
    ) -> None:

        super().__init__(loss, *args, batch_aug=batch_aug, **kwargs)

        self.cache_labels_ = cache_labels
        self.cached_true_train, self.cached_pred_train = None, None
        self.cached_true_val, self.cached_pred_val = None, None

    def forward(self, model, x, y, mixed_precision, training):
        logits, y_pred, y_true = self.feed_model(model, x, y, mixed_precision)
        if self.cache_labels_:
            self.cache_labels(logits, y_true.to(logits.device), training=model.training)

        return self.loss(logits, y_pred), self.calc_metric(
            logits.clone().detach(), y_true
        )

    def calc_metric(self, logits, y_true):
        if logits is None and y_true is None:
            return T.tensor(0.0)
        else:
            return self.calc_acc(logits, y_true)

    def calc_acc(self, logits, y_true):
        y_pred = T.argmax(logits, dim=-1)
        correct = (y_pred == y_true).float().sum()
        return correct / y_true.size(0)

    def calc_precision(self, logits, y_true):
        y_pred = T.argmax(logits, dim=1)
        true_positives = ((y_pred == 1) & (y_true == 1)).float().sum()
        false_positives = ((y_pred == 1) & (y_true == 0)).float().sum()

        if (true_positives + false_positives) == 0:
            return T.tensor(
                1.0
            )  # Return 1 if there are no predicted positives to avoid division by zero

        precision = true_positives / (true_positives + false_positives)
        return precision

    def cache_labels(self, logits, y_true, training):
        if self.cache_labels_:
            y_pred = T.argmax(logits, dim=-1)
            if training:
                if self.cached_true_train is not None:
                    self.cached_true_train = T.cat((self.cached_true_train, y_true))
                    self.cached_pred_train = T.cat((self.cached_pred_train, y_pred))
                else:
                    self.cached_true_train = y_true
                    self.cached_pred_train = y_pred
            else:
                if self.cached_true_val is not None:
                    self.cached_true_val = T.cat((self.cached_true_val, y_true))
                    self.cached_pred_val = T.cat((self.cached_pred_val, y_pred))
                else:
                    self.cached_true_val = y_true
                    self.cached_pred_val = y_pred

    # FIXME USE ONE CACHE AND CLEAR AFTER TRAINING
    def clear_cache(self, training):
        if training:
            self.cached_true_train, self.cached_pred_train = None, None
        else:
            self.cached_true_val, self.cached_pred_val = None, None

    def get_epoch_metrics(self, training, world_size=0):
        metric_dict = None
        if self.cache_labels_:
            if training:
                y_true = (
                    gather_concat(self.cached_true_train, world_size)
                    .cpu()
                    .detach()
                    .numpy()
                )
                y_pred = (
                    gather_concat(self.cached_pred_train, world_size)
                    .cpu()
                    .detach()
                    .numpy()
                )

                metric_dict = model_eval(y_true, y_pred, label="train")
                self.clear_cache(True)
            else:
                y_true = gather_concat(self.cached_true_val, world_size)
                y_pred = gather_concat(self.cached_pred_val, world_size)

                if y_true is not None:
                    y_true = y_true.cpu().detach().numpy()
                    y_pred = y_pred.cpu().detach().numpy()
                    metric_dict = model_eval(y_true, y_pred, label="val")
                    self.clear_cache(False)

        return metric_dict if metric_dict is not None else {}


class CrossEntropyLoss(SupervisedLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, loss=nn.CrossEntropyLoss, **kwargs)


class FineTuneLoss(CrossEntropyLoss):
    def forward(self, model, x, y, mixed_precision):
        x, y, y_true = self.get_model_input(model, x, y, mixed_precision)
        logits = model(x)
        logits = model.forward_cls(logits)
        return self.loss(logits, y), self.calc_metric(logits.clone().detach(), y_true)


class BinaryCrossEntropyLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            loss=nn.BCEWithLogitsLoss,
            **kwargs,
        )

    def calc_metric(self, logits, y_true):
        y_pred = logits > 0.5
        return T.sum(y_pred == y_true) / y_true.size(0)

    def forward(self, model, x, y, *args, **kwargs):
        y[y > 1] = 1
        y = y.float()
        return super().forward(model, x, y, *args, **kwargs)
