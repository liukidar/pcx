import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.utils as pxu
import pcx.functional as pxf
from omegaconf import OmegaConf
# stune
import stune
import json
import copy
import time


class VGG5Model(pxc.EnergyModule):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        act_fn: Callable[[jax.Array], jax.Array]
    ) -> None:
        super().__init__()

        self.act_fn = px.static(act_fn)
        
        # VGG-5 architecture from table:
        # Channel Sizes: [128, 256, 512, 512]
        # Kernel Sizes: [3, 3, 3, 3]
        # Strides: [1, 1, 1, 1]
        # Paddings: [1, 1, 1, 0]
        # Pool: 2×2 with stride 2
        
        self.conv_blocks = [
            # Block 1: input_channels -> 128
            (pxnn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
             pxnn.MaxPool2d(kernel_size=2, stride=2)),
            # Block 2: 128 -> 256
            (pxnn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
             pxnn.MaxPool2d(kernel_size=2, stride=2)),
            # Block 3: 256 -> 512
            (pxnn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
             pxnn.MaxPool2d(kernel_size=2, stride=2)),
            # Block 4: 512 -> 512 with padding=0
            (pxnn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
             pxnn.MaxPool2d(kernel_size=2, stride=2)),
        ]
        
        # Vodes for each conv block (4 vodes for conv layers)
        self.vodes = [pxc.Vode() for _ in range(len(self.conv_blocks))]
        
        # After all conv+pool operations on 32×32 input:
        # 32 -> pool -> 16 -> pool -> 8 -> pool -> 4 -> conv(p=0) -> 2 -> pool -> 1
        # Final feature map: 512 × 1 × 1 = 512
        
        # Classifier head
        self.classifier = pxnn.Linear(512, output_dim)
        
        # Use cross-entropy energy for classification
        self.classifier_vode = pxc.Vode(pxc.ce_energy)
        self.classifier_vode.h.frozen = True

    def __call__(self, x, y, beta=1.0):
        # Process through conv blocks
        for (conv, pool), vode in zip(self.conv_blocks, self.vodes):
            x = conv(x)
            x = self.act_fn(x)
            x = pool(x)
            x = vode(x)
        
        # Flatten for classifier
        x = x.flatten()
        
        # Classifier
        x = self.classifier(x)
        x = self.classifier_vode(x)

        if y is not None:
            # Nudging
            self.classifier_vode.set("h", self.classifier_vode.get("u") - beta * (self.classifier_vode.get("u") - y))
           
        return self.classifier_vode.get("u")


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


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
    # Standard CIFAR-10 data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        lambda x: x.numpy()
    ])
    
    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        lambda x: x.numpy()
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        "~/tmp/cifar10/",
        transform=train_transform,
        download=True,
        train=True,
    )

    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        "~/tmp/cifar10/",
        transform=test_transform,
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


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0, axis_name="batch")
def forward(x, y, *, model: VGG5Model, beta=1.0):
    return model(x, y, beta=beta)


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: VGG5Model):
    y_ = model(x, None)
    return jax.lax.psum(model.energy(), "batch"), y_


@pxf.jit(static_argnums=0, donate_argnames=("model", "optim"))
def train_on_batch(T: int, x: jax.Array, y: jax.Array, *, model: VGG5Model, optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float = 1.0):
    model.train()

    # Init step
    with pxu.step(model, (pxc.STATUS.INIT, None), clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model, beta=beta)
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to(([False, True])), has_aux=True)(
                energy
            )(x, model=model)

        optim_h.step(model, g["model"], True)
    optim_h.clear()

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)(x, model=model)
    optim_w.step(model, g["model"], scale_by=1.0 / x.shape[0])


@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: VGG5Model):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache | pxc.VodeParam):
        y_ = forward(x, None, model=model).argmax(axis=-1)

    return (y_ == y).mean(), y_


def train(dl, T, *, model: VGG5Model, optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float = 1.0):
    
    for i, (x, y) in enumerate(dl):
        # CIFAR-10 images are channels-first format from PyTorch: (B, C, H, W)
        train_on_batch(
            T, x, jax.nn.one_hot(y, 10), model=model, optim_w=optim_w, optim_h=optim_h, beta=beta
        )


def eval(dl, *, model: VGG5Model):
    acc = []
    ys_ = []

    for x, y in dl:
        a, y_ = eval_on_batch(x, y, model=model)
        acc.append(a)
        ys_.append(y_)

    return np.mean(acc), np.concatenate(ys_)


def main(run_info: stune.RunInfo):
    batch_size = run_info["hp/batch_size"]
    nm_epochs = run_info["hp/epochs"]

    model = VGG5Model(
        input_channels=3,  # RGB images
        output_dim=10,     # CIFAR-10 has 10 classes
        act_fn=getattr(jax.nn, run_info["hp/act_fn"]))

    train_dataloader, test_dataloader = get_dataloaders(batch_size)

    # Initialize model with a dummy batch
    dummy_input = jnp.zeros((batch_size, 3, 32, 32))
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(dummy_input, None, model=model)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=run_info["hp/optim/w/lr"],
        peak_value=1.1 * run_info["hp/optim/w/lr"],
        warmup_steps=0.1 * len(train_dataloader) * nm_epochs,
        decay_steps=len(train_dataloader)*nm_epochs,
        end_value=0.1 * run_info["hp/optim/w/lr"],
        exponent=1.0)

    # Wrap optimizers in lambda functions
    optim_h = pxu.Optim(
        lambda: optax.chain(
            optax.sgd(run_info["hp/optim/x/lr"], momentum=run_info["hp/optim/x/momentum"]),
        ),
    )
    optim_w = pxu.Optim(
        lambda: optax.adamw(schedule, weight_decay=run_info["hp/optim/w/wd"]), 
        pxu.M(pxnn.LayerParam)(model)
    )
    
    
    best_accuracy = 0
    accuracies = []
    beta = 1.0
    epoch_times = []

    t_start = time.perf_counter()
    for e in range(nm_epochs):
        t0 = time.perf_counter()

        train(train_dataloader, T=run_info["hp/T"], model=model, optim_w=optim_w, optim_h=optim_h, beta=beta)
        a, y = eval(test_dataloader, model=model)
        accuracies.append(float(a))

        
        epoch_time = time.perf_counter() - t0

        if e > 1:
            epoch_times.append(epoch_time)

        if a > best_accuracy:
            best_accuracy = a
        #print(f"Epoch {e+1}/{nm_epochs}, Accuracy: {a:.4f}, Epoch time: {epoch_time:.2f}s")

    total_time = time.perf_counter() - t_start
    print(f"T={run_info['hp/T']}; VGG-5; Average epoch time: {np.mean(epoch_times):.2f}s, Best accuracy: {best_accuracy:.4f}")

    del train_dataloader
    del test_dataloader

    return float(best_accuracy), accuracies



if __name__ == "__main__":

    for T in [10, 20, 30, 40]:

        run_info={
            "hp/act_fn": "gelu",
            "hp/batch_size": 128,
            "hp/epochs": 10,
            "hp/T": T,
            "hp/beta": 1.0,
            "hp/optim/w/lr": 0.0002968930522737348,
            "hp/optim/w/wd": 0.0003550241114984682,
            "hp/optim/x/lr": 0.010534787245955935,
            "hp/optim/x/momentum": 0.65,
        }

        main(run_info)