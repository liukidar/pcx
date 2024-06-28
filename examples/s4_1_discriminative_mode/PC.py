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
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.utils as pxu
import pcax.functional as pxf
from omegacli import OmegaConf

import json

import random
import sys

from models import get_model
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

@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model , beta=1.0):
    return model(x, y, beta=beta)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model ):
    y_ = model(x, None)
    return jax.lax.pmean(model.energy().sum(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch(T: int, x: jax.Array, y: jax.Array, *, model , optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float = 1.0):
    model.train()

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model, beta=beta)
    optim_h.init(pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True))(model))


    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]), has_aux=True)(
                energy
            )(x, model=model)

        optim_h.step(model, g["model"], True)
    optim_h.clear()

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache | pxc.VodeParam):
        _, g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(energy)(x, model=model)
    optim_w.step(model, g["model"], mul=1/beta)
    
    
@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        outputs = forward(x, None, model=model)
    top1_pred = outputs.argmax(axis=-1)
    top5_indices = jax.lax.top_k(outputs, k=5)[1]

    top1_acc = (top1_pred == y).mean()
    top5_acc = jnp.any(top5_indices == y[:, None], axis=-1).mean()

    return top1_acc, top5_acc, top1_pred


def train(dl, T, *, model , optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float = 1.0):
    for i, (x, y) in enumerate(dl):
        train_on_batch(
            T, x, jax.nn.one_hot(y, model.nm_classes.get()), model=model, optim_w=optim_w, optim_h=optim_h, beta=beta
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

    optim_h = pxu.Optim(
        optax.chain(
            optax.sgd(run_info["hp/optim/x/lr"], momentum=run_info["hp/optim/x/momentum"]),
        )
    )
    optim_w = pxu.Optim(optax.adamw(schedule, weight_decay=run_info["hp/optim/w/wd"]), pxu.Mask(pxnn.LayerParam)(model))
    
    best_accuracy = 0
    best_accuracy5 = 0
    accuracies = []
    accuracies5 = []
    if run_info["hp/beta_factor"] == 0:
            with open(sys.argv[1][:-5] + '_beta.json', 'r') as file:
                betalist = json.load(file)
                # print(betalist)
    below_times = 0
    for e in range(nm_epochs):
        if run_info["hp/beta_factor"] == 0:
            beta = betalist[e]
        else:
            beta = run_info["hp/beta_factor"] * (run_info["hp/beta"] + run_info["hp/beta_ir"]*e)

        if beta >= 1.0:
            beta = 1.0
        elif beta <= -1.0 and run_info["hp/beta_factor"] == 0:
            beta = -1.0
        elif beta <= -1.0 and run_info["hp/beta_factor"] != 0:
            beta = 1.0
        else:
            pass

        train(train_dataloader, T=run_info["hp/T"], model=model, optim_w=optim_w, optim_h=optim_h, beta=beta)
        a, a5, y = eval(test_dataloader, model=model)
        print(a, a5)
        if a < 0.1:
            below_times += 1
        else:
            below_times = 0
        
        if a > best_accuracy:
            best_accuracy = a
        if a5 > best_accuracy5:
            best_accuracy5 = a5
        accuracies.append(float(a))
        accuracies5.append(float(a5))
        if below_times >= 5:
            break

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
