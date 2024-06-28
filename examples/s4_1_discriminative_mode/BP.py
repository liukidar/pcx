from typing import Callable
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os

# Core dependencies
import jax
import jax.numpy as jnp
import optax

# pcax
import pcax as px
import pcax.nn as pxnn
import pcax.utils as pxu
import pcax.functional as pxf
from omegacli import OmegaConf

import json

def se_loss(output, one_hot_label):
    return (jnp.square(output - one_hot_label)).sum()


def ce_loss(output, one_hot_label):
    return -(one_hot_label * jax.nn.log_softmax(output)).sum()

import random
import sys

from models_BP import get_model
from datasets import get_loader

def get_datasetinfo(dataset):
    if dataset == "MNIST":
        return 10, 28
    elif dataset == "FashionMNIST":
        return 10, 28
    elif dataset == "CIFAR10":
        return 10, 32
    elif dataset == "CIFAR100":
        return 100, 32
    elif dataset == "TinyImageNet":
        return 200, 56
    else:
        raise ValueError("Invalid dataset name")

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


@pxf.vmap({"model": None}, in_axes=0, out_axes=0)
def forward(x, *, model):
    return model(x)


@pxf.vmap({"model": None}, in_axes=(0, 0), out_axes=(None, 0), axis_name="batch")
def loss(x, y, *, model):
    y_ = model(x)
    if int(model.se_flag.get()) == 1:
        return jax.lax.pmean(se_loss(y_, y), "batch"), y_
    else:
        return jax.lax.pmean(ce_loss(y_, y), "batch"), y_


@pxf.jit()
def train_on_batch(x: jax.Array, y: jax.Array, *, model, optim_w: pxu.Optim):
    model.train()
    # Learning step
    with pxu.step(model):
        _, g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(loss)(x, y, model=model)
    optim_w.step(model, g["model"])
    
    
@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model):
    model.eval()

    with pxu.step(model):
        outputs = forward(x, model=model)
    top1_pred = outputs.argmax(axis=-1)
    top5_indices = jax.lax.top_k(outputs, k=5)[1]

    top1_acc = (top1_pred == y).mean()
    top5_acc = jnp.any(top5_indices == y[:, None], axis=-1).mean()

    return top1_acc, top5_acc, top1_pred


def train(dl, *, model , optim_w: pxu.Optim):
    for i, (x, y) in enumerate(dl):
        train_on_batch(
             x, jax.nn.one_hot(y, model.nm_classes.get()), model=model, optim_w=optim_w
        )


def eval(dl, *, model):
    acc = []
    acc5 = []
    ys_ = []

    for x, y in dl:
        a, a5, y_ = eval_on_batch(x, y, model=model)
        acc.append(a)
        acc5.append(a5)
        ys_.append(y_)

    return np.mean(acc), np.mean(acc5), np.concatenate(ys_)


def main(run_info):

   
    dataset_name = run_info["hp/dataset"]

    batch_size = run_info["hp/batch_size"]
    nm_epochs = run_info["hp/epochs"]

    nm_classes, input_size = get_datasetinfo(dataset_name)

    model = get_model(
        model_name=run_info["hp/model"], 
        nm_classes=nm_classes, 
        act_fn=getattr(jax.nn, run_info["hp/act_fn"]),
        input_size=input_size,
        se_flag=run_info["hp/se_flag"])
    
    train_dataloader, test_dataloader = get_loader(dataset_name, batch_size)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=run_info["hp/optim/w/lr"],
        peak_value=1.1 * run_info["hp/optim/w/lr"],
        warmup_steps=0.1 * len(train_dataloader) * nm_epochs,
        decay_steps=len(train_dataloader)*nm_epochs,
        end_value=0.1 * run_info["hp/optim/w/lr"],
        exponent=1.0)

    optim_w = pxu.Optim(optax.adamw(schedule, weight_decay=run_info["hp/optim/w/wd"]), pxu.Mask(pxnn.LayerParam)(model))
    
    best_accuracy = 0
    best_accuracy5 = 0
    accuracies = []
    accuracies5 = []
    for e in range(nm_epochs):
        train(train_dataloader, model=model, optim_w=optim_w)
        a, a5, y = eval(test_dataloader, model=model)
        if a > best_accuracy:
            best_accuracy = a
        if a5 > best_accuracy5:
            best_accuracy5 = a5
        accuracies.append(float(a))
        accuracies5.append(float(a5))

    del train_dataloader
    del test_dataloader

    return float(best_accuracy), float(best_accuracy5), accuracies, accuracies5


if __name__ == "__main__":
    import os
    import sys
    import seed5 as seed
    run_info = seed.RunInfo(
        OmegaConf.load(sys.argv[1])
    )
    seed.run(main)(run_info)
