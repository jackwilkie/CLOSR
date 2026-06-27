#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:01:56 2023

@author: jack
"""

import torch as T
import torch.nn as nn
import time
import copy
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import shutil
from typing import Optional
import tracemalloc
import os


# checkpointing
def ensure_extension(filename, extension):
    if not filename.endswith(extension):
        filename += extension
    return filename


def convert_distributed_to_local(model_state_dict):
    return {
        key.replace("module.", "", 1): value for key, value in model_state_dict.items()
    }


def make_checkpoint(
    model,
    optimiser=None,
    schedular=None,
    scalar=None,
    stats=None,
    logger=None,
    path: str = "checkpoint.pt.tar",
    rank: int = None,
    **kwargs,
):

    if rank is None or rank == 0:
        path = ensure_extension(path, ".pt.tar")

        checkpoint = {
            "model_state_dict": model.state_dict()
            if not isinstance(model, DDP)
            else model.module.state_dict(),
            "optimiser_state_dict": None if not optimiser else optimiser.state_dict(),
            "scalar_state_dict": None if not scalar else scalar.state_dict(),
            "schedular_state_dict": None if not schedular else schedular.state_dict(),
            "schedular": schedular,
            "stats": stats,
            "log": None if not logger else logger.get_log(),
        }

        if kwargs:
            checkpoint = {**checkpoint, **kwargs}
        T.save(checkpoint, path)


# distributed training
def setup(rank: int, world_size: int) -> None:
    """
    Initalises nccl backend for distributed training communication/
    nccl allows for fast communication of GPU tensors

    args:
        rank (int): rank of process calling function
        world_size (int): Number of processes used for distributed training

    returns:
        None
    """
    if world_size > 0:
        os.environ["MASTER_ADDR"] = (
            "127.0.0.1"  # Set to the IP address of the master node
        )
        os.environ["MASTER_PORT"] = "29500"  # Set to an unused port

        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )


def cleanup() -> None:
    dist.destroy_process_group()  # closes distributed backend


def is_dist_avail_and_initalised() -> bool:
    return (
        True if dist.is_available() and dist.is_initialized() else False
    )  # checks if code is currently being ran distributed


def get_rank() -> int:
    return (
        0 if not is_dist_avail_and_initalised() else dist.get_rank()
    )  # get rank of current process


def is_main_process() -> bool:
    return get_rank() == 0  # main process is typically assinged rank 0


def gather_mean(metric, world_size):
    if world_size is None or world_size == 0:
        return metric
    else:
        dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        return metric / world_size


def gather_concat(tensor, world_size):
    if world_size is None or world_size == 0:
        return tensor
    else:
        empty_tensor = [T.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(empty_tensor, tensor)
        filled = T.cat(empty_tensor, dim=0)
        return filled


# custom class for loss calculation
class LossCalc:
    """
    Class for calculating loss, must have call method which accepts model
    .and batch and returns loss tensor
    """

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, model, batch):
        x, y = batch
        z = model(x)
        return self.loss_fn(z, y)


def line_print(msg, line_width=None):
    if line_width is None:
        line_width, _ = shutil.get_terminal_size()

    padding = padding = " " * (line_width - len(msg))
    msg = msg + padding
    msg = msg[:line_width]
    print(msg, end="\r")


# custom training loop
def train(
    model: nn.Module,
    optimiser,
    loss_calc: LossCalc,
    epochs: int,
    train_dl: T.utils.data.DataLoader,
    val_dl: T.utils.data.DataLoader = None,
    ep_log_interval: int = 1,
    scheduler=None,
    logger=None,
    eval_func=None,
    eval_interval: int = 1,
    scalar=None,
    print_batch_loss: bool = False,
    checkpoint_interval: int = 0,
    checkpoint_path: str = "checkpoints/checkpoint",
    max_batches=0,
    grad_accumulation: int = 1,
    start_epoch: int = 0,
    rank=None,
    world_size=None,
    prefix="",
    val_pass=True,
    save_best: Optional[str] = None,
    grad_clipping: Optional[float] = None,
) -> dict:
    """
    Custom Pytorch training loop

    Parameters
    ----------
    model : nn.Module
        Pytorch model to train.
    optimiser :
        Pytorch optimiser.
    loss_calc : LossCalc
        Method which takes model and batch and returns loss tensor.
    epochs : int
        Number of epochs to train model for.
    train_dl : T.utils.data.DataLoader
        Train dataloader.
    val_dl : T.utils.data.DataLoader, optional
        Validation dataloader. The default is None.
    ep_log_interval : int, optional
        Frequency, in epochs, to display verbose. Doesnt display if 0. The default is 1.
    scheduler :  optional
        Learning rate scheduler. The default is None.
    logger : optional
        Logger to pass training stats to if specified. Requires log_data function, stats will be passed to this as a dict.
    eval_func:
        Function which accepts model, train_dl, val_dl, and returns model performance stats to log.
    eval_interval: int
        How often in epochs to ecaluate them model using eval_func.
    scalar:
        Pytorch gradient scalar, will use mixed precision if provided otherwise will use full precision.
    print_batch_loss: bool
        Print loss after each batch if True. The default is False.
    checkpoint_interval: int
        Period to wait (in epochs) before making a new model checkpoint.
    checkpoint_path: str
        Path to directory to file to save checkoints. _epoch_{epoch_num}.pt.tar will be appended before saving checkpoint.
    max_batchs: int
        Maximum number of batches per epoch. The default is 0 (uses entire dataloader).
    Returns
    -------
    history : dict
        Dictionary containing loss metrics.

    """
    # collect training and validation metrics for each epoch
    tracemalloc.start()
    history = []
    start_time = time.time()
    print(loss_calc)
    world_size = world_size or 0
    loss_scalar = world_size if world_size > 1 else 1
    rank = rank or 0
    grad_accumulation = (
        1 if grad_accumulation == 0 else grad_accumulation
    )  # prevent modulo by 0 errors
    best_loss = None

    if world_size > 0:
        dist.barrier()

    if scalar is not None:
        mixed_precision = True
    else:
        mixed_precision = False

    optimiser.zero_grad()
    grad_counter = 0  # init gradient accumulation counter

    # start training loop
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()

        # shuffle which data is on each gpu each epoch for distributed training
        if isinstance(train_dl.sampler, DistributedSampler):
            train_dl.sampler.set_epoch(epoch)

        if val_dl is not None:
            if isinstance(val_dl.sampler, DistributedSampler):
                val_dl.sampler.set_epoch(epoch)

        epoch_num = epoch + 1
        train_acc = 0

        # --------------- train and evaluate on training dataset ---------------

        train_loss = 0.0  # initalise train loss

        model.train()  # set model to train
        if rank == 0:
            print(f"Training epoch {epoch}/{epochs}:")

        for batch_num, batch in enumerate(train_dl):
            # print(f'batch {batch_num}/{len(train_dl)}', end = '\r')

            with T.cuda.amp.autocast(enabled=mixed_precision):
                grad_counter += 1  # increment gradient accumlation counter
                x, y = batch  # get batch
                loss, acc = loss_calc(
                    model, x, y, mixed_precision=mixed_precision, training=True
                )  # compute training loss

                batch_acc = acc  # .item()

                if isinstance(loss, tuple):
                    batch_loss = loss[0].clone().detach()  # .item()
                    loss = sum(loss)
                else:
                    batch_loss = loss.clone().detach()  # .item()

            # apply backpropogation
            if mixed_precision:  # scale grads if using mixed precision
                scalar.scale(loss).backward()  # scale grads to prevent underflow

                if grad_counter % grad_accumulation == 0:
                    # print('updating weights, mixed precision')
                    if grad_clipping is not None:
                        T.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)
                    scalar.step(optimiser)  # reverse scaling for weight updates
                    scalar.update()  # update scalar state
                    optimiser.zero_grad()

                    if (
                        scheduler is not None
                    ):  # step learning rate if a schedular is provided
                        scheduler.step()
            else:
                loss.backward()  # compute gradients

                # update gradient after gradient accumlation interval
                if grad_counter % grad_accumulation == 0:
                    # print('updating weights, full precision')
                    if grad_clipping is not None:
                        T.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

                    optimiser.step()  # update weights
                    optimiser.zero_grad()

                    if (
                        scheduler is not None
                    ):  # step learning rate if a schedular is provided
                        scheduler.step()

            if print_batch_loss and rank == 0:
                line_print(
                    f"Train batch {batch_num}/{len(train_dl)}  |  train_loss: {batch_loss}  |  Train acc: {batch_acc}  |  Time per batch: {(time.time() - epoch_start_time) / (float(batch_num) + 1)}"
                )

            batch_loss *= y.size(0)
            train_loss += (
                batch_loss  # multiply sample loss by batch size for batch loss
            )
            train_acc += batch_acc

            if batch_num + 1 == max_batches:
                break

        train_loss = (loss_scalar * train_loss) / len(
            train_dl.dataset
        )  # find per sample loss
        train_acc = train_acc / (len(train_dl) + 1e-6)

        # -------------------- evaluate on validation dataset --------------------

        val_loss = 0.0
        val_acc = 0.0

        model.eval()  # set model to evaluat

        if val_dl is not None and val_pass:
            if rank == 0:
                print(f"\nValidation epoch {epoch}/{epochs}:")
            with T.no_grad():
                for batch_num, batch in enumerate(val_dl):
                    x, y = batch  # get batch

                    # calculate validation loss using mixed precision
                    with T.cuda.amp.autocast(enabled=mixed_precision):
                        loss, acc = loss_calc(
                            model, x, y, mixed_precision=mixed_precision, training=False
                        )  # compute validation loss

                    batch_acc = acc  # .item()
                    if isinstance(loss, tuple):
                        batch_loss = loss[0].clone().detach()  # .item()
                        # loss = sum(loss)
                    else:
                        batch_loss = loss.clone().detach()  # .item()

                    # batch_loss = loss#.item()
                    if print_batch_loss and rank == 0:
                        line_print(
                            f"Val batch {batch_num}/{len(val_dl)}, val_loss: {batch_loss}  |  val acc: {batch_acc}"
                        )

                    batch_loss *= y.size(0)
                    val_loss += (
                        batch_loss  # multiply sample loss by batch size for batch loss
                    )
                    val_acc += batch_acc

                val_loss = (loss_scalar * val_loss) / len(val_dl.dataset)
                val_acc = val_acc / len(val_dl)

        # -- gather performance metrics across processes
        train_loss = gather_mean(train_loss, world_size)
        train_acc = gather_mean(train_acc, world_size)

        epoch_metrics = {
            "epoch": epoch_num,
            "train_loss": round(train_loss.item(), 6),
            "train_time": time.time() - start_time,
            "train_acc": train_acc.item(),
        }

        epoch_metrics = {
            **epoch_metrics,
            **loss_calc.get_epoch_metrics(training=True, world_size=world_size),
            **loss_calc.get_epoch_metrics(training=False, world_size=world_size),
        }

        if val_pass and val_dl is not None:
            val_loss = gather_mean(val_loss, world_size)
            val_acc = gather_mean(val_acc, world_size)
            epoch_metrics["val_loss"] = round(val_loss.item(), 6)
            epoch_metrics["val_acc"] = val_acc.item()

            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                if save_best is not None:
                    make_checkpoint(
                        model=model,
                        optimiser=optimiser,
                        schedular=scheduler,
                        stats=epoch_metrics,
                        scalar=scalar,
                        logger=logger,
                        path=save_best,
                    )

        # -- calculate evaluation metrics periodically
        if (
            eval_func is not None
            and eval_interval != 0
            and epoch_num % eval_interval == 0
            and val_dl
        ):
            with T.no_grad():
                epoch_metrics = {
                    **epoch_metrics,
                    **eval_func(
                        model,
                        train_dl,
                        val_dl,
                        mixed_precision=mixed_precision,
                        world_size=world_size,
                    ),
                }

        epoch_metrics = {
            f"{prefix}{key}": value for key, value in epoch_metrics.items()
        }
        history.append(copy.deepcopy(epoch_metrics))

        if world_size == 0 or rank == 0:
            # -- log model performance
            if logger is not None:
                print("\n")
                logger.log_data(epoch_metrics)

            # -- make model checkpoint periodically
            if checkpoint_interval != 0 and epoch_num % checkpoint_interval == 0:
                make_checkpoint(
                    model=model,
                    optimiser=optimiser,
                    schedular=scheduler,
                    stats=epoch_metrics,
                    scalar=scalar,
                    logger=logger,
                    path=f"{checkpoint_path}_epoch_{epoch_num}",
                )

            # -- periodically print performance to screen
            if ep_log_interval != 0 and epoch_num % ep_log_interval == 0:
                print(" | ".join(f"{k}: {v}" for k, v in epoch_metrics.items()))

        if world_size > 0:
            dist.barrier()

    return history


# custom training loop
def train_no_dl(
    model: nn.Module,
    optimiser,
    loss_calc: LossCalc,
    epochs: int,
    x_train,
    y_train,
    x_val,
    y_val,
    ep_log_interval: int = 1,
    scheduler=None,
    logger=None,
    eval_func=None,
    eval_interval: int = 1,
    checkpoint_interval: int = 0,
    checkpoint_path: str = "checkpoints/checkpoint",
    grad_accumulation: int = 1,
    start_epoch: int = 0,
    prefix="",
    val_pass=True,
):
    """
    Training loop for models which train on entire dataset via batch descent,
    reduces overhead by removing dataloader

    """

    # collect training and validation metrics for each epoch
    history = []
    start_time = time.time()
    optimiser.zero_grad()
    grad_counter = 0

    # start training loop
    for epoch in range(start_epoch, epochs):
        epoch_num = epoch + 1
        train_acc = 0

        # --------------- train and evaluate on training dataset ---------------

        train_loss = 0.0  # initalise train loss

        model.train()  # set model to train

        grad_counter += 1  # increment gradient accumlation counter

        loss, acc = loss_calc(
            model, x_train, y_train, mixed_precision=False, training=True
        )  # compute training loss

        batch_acc = acc  # .item()

        if isinstance(loss, tuple):
            batch_loss = loss[0].clone().detach()  # .item()
            loss = sum(loss)
        else:
            batch_loss = loss.clone().detach()  # .item()

        loss.backward()  # compute gradients

        # update gradient after gradient accumlation interval
        if grad_counter % grad_accumulation == 0:
            # print('updating weights, full precision')
            optimiser.step()  # update weights
            optimiser.zero_grad()

            if scheduler is not None:  # step learning rate if a schedular is provided
                scheduler.step()

        train_loss = batch_loss  # multiply sample loss by batch size for batch loss
        train_acc = batch_acc

        # -------------------- evaluate on validation dataset --------------------

        if val_pass:
            val_loss = 0  # init val loss
            val_acc = 0.0

            model.eval()  # set model to evaluat

            with T.no_grad():
                # calculate validation loss using mixed precision
                loss, acc = loss_calc(
                    model, x_val, y_val, mixed_precision=False, training=False
                )  # compute validation loss

                batch_acc = acc  # .item()
                if isinstance(loss, tuple):
                    batch_loss = loss[0].clone().detach()  # .item()
                    # loss = sum(loss)
                else:
                    batch_loss = loss.clone().detach()  # .item()

                # batch_loss = loss#.item()
                val_loss = (
                    batch_loss  # multiply sample loss by batch size for batch loss
                )
                val_acc = batch_acc

        # -- gather performance metrics across processes
        epoch_metrics = {
            "epoch": epoch_num,
            "train_loss": round(train_loss.item(), 6),
            "train_time": time.time() - start_time,
            "train_acc": train_acc.item(),
        }

        epoch_metrics = {
            **epoch_metrics,
            **loss_calc.get_epoch_metrics(training=True, world_size=0),
            **loss_calc.get_epoch_metrics(training=False, world_size=0),
        }

        if val_pass:
            epoch_metrics["val_loss"] = round(val_loss.item(), 6) if val_pass else 0.0
            epoch_metrics["val_acc"] = val_acc.item()

        # -- calculate evaluation metrics periodically
        if (
            eval_func is not None
            and eval_interval != 0
            and epoch_num % eval_interval == 0
            and x_val
        ):
            epoch_metrics = {
                **epoch_metrics,
                **eval_func(
                    model,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                ),
            }

        epoch_metrics = {
            f"{prefix}{key}": value for key, value in epoch_metrics.items()
        }
        history.append(copy.deepcopy(epoch_metrics))

        # -- log model performance
        if logger is not None:
            print("\n")
            logger.log_data(epoch_metrics)

        # -- make model checkpoint periodically
        if checkpoint_interval != 0 and epoch_num % checkpoint_interval == 0:
            make_checkpoint(
                model=model,
                optimiser=optimiser,
                schedular=scheduler,
                stats=epoch_metrics,
                logger=logger,
                path=f"{checkpoint_path}_epoch_{epoch_num}",
            )

        # -- periodically print performance to screen
        if ep_log_interval != 0 and epoch_num % ep_log_interval == 0:
            print(" | ".join(f"{k}: {v}" for k, v in epoch_metrics.items()))

    return history
