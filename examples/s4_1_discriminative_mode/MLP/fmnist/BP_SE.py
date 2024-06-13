from typing import Callable
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms


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
import copy

def se_loss(output, one_hot_label):
    return (jnp.square(output - one_hot_label)).sum()


def ce_loss(output, one_hot_label):
    return -(one_hot_label * jax.nn.log_softmax(output)).sum()


# We create our model, which inherits from pxc.EnergyModule, so to have access to the notion
# energy. The constructor takes in input all the hyperparameters of the model. Being static
# values, if we intend to save them withing the model we must wrap them into a 'StaticParam'.
class LinearModel(px.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nm_layers: int,
        act_fn: Callable[[jax.Array], jax.Array]
    ) -> None:
        super().__init__()

        self.act_fn = px.static(act_fn)
        
        self.layers = [pxnn.Linear(input_dim, hidden_dim)] + [
            pxnn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 2)
        ] + [pxnn.Linear(hidden_dim, output_dim)]

        # the default ruleset for a Vode is: `{"STATUS.INIT": ("h, u <- u",),}` which means:
        # "if the status is set to 'STATUS.INIT', everytime I set 'u', save that value not only
        # in 'u', but also in 'x', which is exactly the behvaiour of a forward pass.
        # By default if not specified, the behaviour is '* <- *', i.e., save everything passed
        # to the vode via __call__ (remember vode(a) equals to vode.set("u", a)).
        #
        # Since we are doing classification, we replace the last energy with the equivalent of
        # cross entropy loss for predictive coding.

    def __call__(self, x):
        for l in self.layers[:-1]:
            
            x = self.act_fn(l(x))

        x = self.layers[-1](x)
           
        return x


# This is a simple collate function that stacks numpy arrays used to interface
# the PyTorch dataloader with JAX. In the future we hope to provide custom dataloaders
# that are independent of PyTorch.


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


# The dataloader assumes cuda is being used, as such it sets 'pin_memory = True' and
# 'prefetch_factor = 2'. Note that the batch size should be constant during training, so
# we set 'drop_last = True' to avoid having to deal with variable batch sizes.
class TorchDataloader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=1,
        pin_memory=True,
        timeout=0,
        worker_init_fn=None,
        persistent_workers=True,
        prefetch_factor=2,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=True if batch_sampler is None else None,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )


def get_dataloaders(batch_size: int):
    t = transforms.Compose([
    transforms.ToTensor(),  # Converts to tensor
    transforms.Normalize((0.5,), (0.5,)),  # Normalize with mean and std dev
    transforms.Lambda(lambda x: torch.flatten(x)),  # Flatten the tensor
    lambda x: x.numpy()
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        "~/tmp/fmnist/",
        transform=t,
        download=True,
        train=True,
    )

    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        "~/tmp/fmnist/",
        transform=t,
        download=True,
        train=False,
    )
        
    test_dataloader = TorchDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_dataloader, test_dataloader


@pxf.vmap({"model": None}, in_axes=0, out_axes=0)
def forward(x, *, model: LinearModel):
    return model(x)


@pxf.vmap({"model": None}, in_axes=(0, 0), out_axes=(None, 0), axis_name="batch")
def loss(x, y, *, model: LinearModel):
    y_ = model(x)
    return jax.lax.pmean(se_loss(y_, y), "batch"), y_


@pxf.jit()
def train_on_batch(x: jax.Array, y: jax.Array, *, model: LinearModel, optim_w: pxu.Optim):
    model.train()

    # Learning step
    with pxu.step(model):
        _, g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(loss)(x, y,  model=model)
    optim_w.step(model, g["model"])

@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: LinearModel):
    model.eval()

    with pxu.step(model):
        y_ = forward(x, model=model).argmax(axis=-1)

    return (y_ == y).mean(), y_


def train(dl, *, model: LinearModel, optim_w: pxu.Optim):
    for i, (x, y) in enumerate(dl):
        train_on_batch(
            x, jax.nn.one_hot(y, 10), model=model, optim_w=optim_w
        )


def eval(dl, *, model: LinearModel):
    acc = []
    ys_ = []

    for x, y in dl:
        a, y_ = eval_on_batch(x, y, model=model)
        acc.append(a)
        ys_.append(y_)

    return np.mean(acc), np.concatenate(ys_)


def main(run_info):
    batch_size = run_info["hp/batch_size"]
    nm_epochs = run_info["hp/epochs"]

    model = LinearModel(
        input_dim=784,
        hidden_dim=128,
        nm_layers=4,
        output_dim=10, 
        act_fn=getattr(jax.nn, run_info["hp/act_fn"]))
    
    train_dataloader, test_dataloader = get_dataloaders(batch_size)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=run_info["hp/optim/w/lr"],  # 初始学习率
        peak_value=1.1 * run_info["hp/optim/w/lr"],  # 最大学习率
        warmup_steps=0.1 * len(train_dataloader) * nm_epochs,  # 预热步数 
        decay_steps=len(train_dataloader)*nm_epochs,  # 衰减步数
        end_value=0.1 * run_info["hp/optim/w/lr"],  # 最小学习率
        exponent=1.0)

    optim_w = pxu.Optim(optax.adamw(schedule,weight_decay=run_info["hp/optim/w/wd"]), pxu.Mask(pxnn.LayerParam)(model))
    
    best_accuracy = 0
    accuracies = []
    for e in range(nm_epochs):
        train(train_dataloader, model=model, optim_w=optim_w)
        a, y = eval(test_dataloader, model=model)
        if a > best_accuracy:
            best_accuracy = a
        accuracies.append(float(a))

    del train_dataloader
    del test_dataloader

    return float(best_accuracy), accuracies


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    import seed
    run_info = seed.RunInfo(
        OmegaConf.load(sys.argv[1])
    )
    seed.run(main)(run_info)
