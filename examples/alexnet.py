"""
Tutorial #4: CIFAR10 via AlexNet

This script demonstrates how to code a PCN based on AlexNet and train it on CIFAR10.
Note: This requires PyTorch for dataset handling and can take significant time to train.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import optax

import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.functional as pxf
import pcx.utils as pxu


# ==============================================================================
# Model Definition
# ==============================================================================

class AlexNet(pxc.EnergyModule):
    """
    AlexNet-style PCN for CIFAR10 classification.
    
    Uses convolutional and fully connected layers with vodes between them
    to implement predictive coding.
    """
    def __init__(
        self,
        nm_classes: int,
        act_fn: Callable[[jax.Array], jax.Array]
    ) -> None:
        super().__init__()

        self.nm_classes = nm_classes
        self.act_fn = px.static(act_fn)

        # Define convolutional feature extraction layers
        self.feature_layers = [
            (
                pxnn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(64, 192, kernel_size=(3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
                self.act_fn
            ),
            (
                pxnn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
                self.act_fn
            ),
            (
                pxnn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            )
        ]
        
        # Define classifier layers
        self.classifier_layers = [
            (
                pxnn.Linear(256 * 2 * 2, 4096),
                self.act_fn
            ),
            (
                pxnn.Linear(4096, 4096),
                self.act_fn
            ),
            (
                pxnn.Linear(4096, self.nm_classes),
            )
        ]

        # Define vodes - one for each block
        self.vodes = [
            pxc.Vode() for _ in self.feature_layers
        ] + [
            pxc.Vode() for _ in self.classifier_layers[:-1]
        ] + [pxc.Vode(pxc.ce_energy)]

        self.vodes[-1].h.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array):
        # Feature extraction
        for block, node in zip(self.feature_layers, self.vodes[:len(self.feature_layers)]):
            for layer in block:
                x = layer(x)
            x = node(x)

        # Flatten for classifier
        x = x.flatten()
        
        # Classification
        for block, node in zip(self.classifier_layers, self.vodes[len(self.feature_layers):]):
            for layer in block:
                x = layer(x)
            x = node(x)

        if y is not None:
            self.vodes[-1].set("h", y)

        return self.vodes[-1].get("u")


# ==============================================================================
# DataLoader
# ==============================================================================

def numpy_collate(batch):
    """Collate function to convert PyTorch tensors to numpy arrays."""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class TorchDataloader(torch.utils.data.DataLoader):
    """Custom DataLoader that returns numpy arrays instead of PyTorch tensors."""
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
    """Create CIFAR10 train and test dataloaders."""
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        "~/tmp/cifar10/",
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

    test_dataset = torchvision.datasets.CIFAR10(
        "~/tmp/cifar10/",
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


# ==============================================================================
# Transform Functions
# ==============================================================================

@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: AlexNet):
    return model(x, y)


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: AlexNet):
    y_ = model(x, None)
    return jax.lax.psum(model.energy(), "batch"), y_


# ==============================================================================
# Training Functions
# ==============================================================================

@pxf.jit(static_argnums=0)
def train_on_batch(
    T: int,
    x: jax.Array,
    y: jax.Array,
    *,
    model: AlexNet,
    optim_w: pxu.Optim,
    optim_h: pxu.Optim
):
    model.train()

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)
    
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    
    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(
                pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
                has_aux=True
            )(energy)(x, model=model)
        
        optim_h.step(model, g["model"])
    
    optim_h.clear()

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)(x, model=model)
    optim_w.step(model, g["model"], scale_by=1.0/x.shape[0])


@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: AlexNet):
    model.eval()
    
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model).argmax(axis=-1)
    
    return (y_ == y).mean(), y_


def train(dl, T, *, model: AlexNet, optim_w: pxu.Optim, optim_h: pxu.Optim):
    for i, (x, y) in enumerate(dl):
        train_on_batch(T, x, jax.nn.one_hot(y, model.nm_classes), model=model, optim_w=optim_w, optim_h=optim_h)
        
        if i % 100 == 0:
            print(f"  Batch {i}/{len(dl)}", flush=True)


def eval(dl, *, model: AlexNet):
    acc = []
    ys_ = []
    
    for x, y in dl:
        a, y_ = eval_on_batch(x, y, model=model)
        acc.append(a)
        ys_.append(y_)
    
    return np.mean(acc), np.concatenate(ys_)


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    print("=" * 80)
    print("Tutorial #4: CIFAR10 via AlexNet")
    print("=" * 80)
    print("\nWARNING: This training can take considerable time!")
    print("Consider reducing nm_epochs for faster experimentation.\n")
    
    # Hyperparameters
    batch_size = 128
    nm_epochs = 10
    T = 13  # Inference steps
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {nm_epochs}")
    print(f"  Inference steps (T): {T}")
    
    # Create model
    model = AlexNet(
        nm_classes=10,
        act_fn=jax.nn.gelu
    )
    print(f"\nModel created:")
    print(f"  Feature layers: {len(model.feature_layers)}")
    print(f"  Classifier layers: {len(model.classifier_layers)}")
    print(f"  Total vodes: {len(model.vodes)}")
    
    # Create optimizers
    optim_h = pxu.Optim(lambda: optax.sgd(5e-2, momentum=0.5, nesterov=True))
    optim_w = pxu.Optim(lambda: optax.adamw(1e-4), pxu.M(pxnn.LayerParam)(model))
    print("Optimizers initialized")
    
    # Load data
    print("\nLoading CIFAR10 dataset...")
    train_dataloader, test_dataloader = get_dataloaders(batch_size)
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    for e in range(nm_epochs):
        print(f"\nEpoch {e + 1}/{nm_epochs}")
        train(train_dataloader, T=T, model=model, optim_w=optim_w, optim_h=optim_h)
        
        print(f"  Evaluating...")
        a, y = eval(test_dataloader, model=model)
        print(f"  Test Accuracy: {a * 100:.2f}%")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()