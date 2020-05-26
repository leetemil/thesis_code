import math
import time
from collections import defaultdict
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from .utils import make_loading_bar, readable_time, eta, get_gradient_norm

def log_progress(epoch, time, progress, total, end, **kwargs):
    report = f"Epoch: {epoch:5} "
    digits = int(math.log10(total)) + 1
    report += f"Time: {readable_time(time)} ETA: {readable_time(eta(time, progress / total))} [{progress:{digits}}/{total}] {make_loading_bar(40, progress / total)}"

    for key, value in kwargs.items():
        if type(value) == int:
            report += (f" {key}: {value:5}")
        elif type(value) == float:
            report += (f" {key}: {value:7.5f}")
        else:
            report += (f" {key}: {value}")
    print(report, end = end)

def train_epoch(epoch, model, optimizer, train_loader, log_interval, clip_grad_norm = None, clip_grad_value = None, scheduler = None, random_weighted_sampling = False):
    """
        epoch: Index of the epoch to run
        model: The model to run data through. Forward should return a tuple of (loss, metrics_dict).
        optimizer: The optimizer to step with at every batch
        train_loader: PyTorch DataLoader to generate batches of training data
        log_interval: Interval in seconds of how often to log training progress (0 to disable batch progress logging)
    """
    progressed_data = 0
    data_len = len(train_loader.dataset)
    num_batches = (data_len // train_loader.batch_size) + 1

    if log_interval != 0:
        log_progress(epoch, 0, progressed_data, data_len, "\r", Loss = 0)
    last_log_time = time.time()

    train_loss = 0
    train_count = 0
    start_time = time.time()

    if scheduler is not None:
        learning_rates = []

    acc_metrics_dict = defaultdict(lambda: 0)

    for batch_idx, xb in enumerate(train_loader):

        batch_size, loss, batch_metrics_dict = train_batch(model, optimizer, xb, clip_grad_norm, clip_grad_value, scheduler, epoch, batch_idx, num_batches, random_weighted_sampling = random_weighted_sampling)

        progressed_data += batch_size

        # Calculate accumulated metrics
        for key, value in batch_metrics_dict.items():
            acc_metrics_dict[key] += value * batch_size
            acc_metrics_dict[key + "_count"] += batch_size
        metrics_dict = {k: acc_metrics_dict[k] / acc_metrics_dict[k + "_count"] for k in acc_metrics_dict.keys() if not k.endswith("_count")}

        train_loss += loss.item() * batch_size
        train_count += batch_size

        if log_interval != 0 and (log_interval == "batch" or time.time() - last_log_time > log_interval):
            last_log_time = time.time()
            log_progress(epoch, time.time() - start_time, progressed_data, data_len, "\r", Loss = train_loss / train_count, **metrics_dict)

        if scheduler is not None:
            learning_rates += scheduler.get_last_lr()

    average_loss = train_loss / train_count
    if log_interval != 0:
        log_progress(epoch, time.time() - start_time, data_len, data_len, "\n", Loss = train_loss / train_count, **metrics_dict)

    if scheduler is not None:
        metrics_dict['learning_rates'] = learning_rates

    return average_loss, metrics_dict

def train_batch(model, optimizer, xb, clip_grad_norm = None, clip_grad_value = None, scheduler = None, epoch = None, batch = None, num_batches = None, random_weighted_sampling = False):
    model.train()
    batch_size = xb.size(0) if isinstance(xb, torch.Tensor) else xb[0].size(0)

    # Reset gradient for next batch
    optimizer.zero_grad()

    if model.training and random_weighted_sampling:
        # this is currently only implemented for the VAE model
        model.weight_loss = False

    # Push whole batch of data through model.forward()
    if isinstance(xb, Tensor):
        loss, batch_metrics_dict = model(xb)
    else:
        loss, batch_metrics_dict = model(*xb)

    # Calculate the gradient of the loss w.r.t. the graph leaves
    loss = loss.mean() # fixes multi GPU issues; leave it
    loss.backward()

    if clip_grad_norm is not None:
        clip_grad_norm_(model.parameters(), clip_grad_norm)
    if clip_grad_value is not None:
        clip_grad_value_(model.parameters(), clip_grad_value)

    # Step in the direction of the gradient
    optimizer.step()

    # Schedule learning rate
    if scheduler is not None:
        assert epoch is not None
        assert batch is not None
        assert num_batches is not None
        scheduler.step(epoch + batch / num_batches)

    return batch_size, loss, batch_metrics_dict

def validate(epoch, model, validation_loader):
    model.eval()

    validation_loss = 0
    validation_count = 0
    with torch.no_grad():
        acc_metrics_dict = defaultdict(lambda: 0)
        for i, xb in enumerate(validation_loader):
            batch_size = xb.size(0) if isinstance(xb, torch.Tensor) else xb[0].size(0)

            # Push whole batch of data through model.forward()
            if isinstance(xb, Tensor):
                loss, batch_metrics_dict = model(xb)
            else:
                loss, batch_metrics_dict = model(*xb)

            loss = loss.mean()

            # Calculate accumulated metrics
            for key, value in batch_metrics_dict.items():
                acc_metrics_dict[key] += value * batch_size
                acc_metrics_dict[key + "_count"] += batch_size

            validation_loss += loss.item() * batch_size
            validation_count += batch_size

        metrics_dict = {k: acc_metrics_dict[k] / acc_metrics_dict[k + "_count"] for k in acc_metrics_dict.keys() if not k.endswith("_count")}
        average_loss = validation_loss / validation_count

    return average_loss, metrics_dict
