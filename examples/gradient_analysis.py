"""
Tutorial #1: Analysis of PCNs

This script analyzes PCN behavior with respect to gradient updates across different
network architectures. It studies how hidden width and learning rate affect training.
"""

from typing import Callable
import random

import jax
import jax.numpy as jnp
import numpy as np

import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.functional as pxf
import pcx.utils as pxu

from sklearn.datasets import make_moons
import optax


# ==============================================================================
# Model Definition
# ==============================================================================

class Model(pxc.EnergyModule):
    """PCN model for gradient analysis."""
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

        self.vodes = [
            pxc.Vode() for _ in range(nm_layers - 1)
        ] + [pxc.Vode(pxc.ce_energy)]
        
        self.vodes[-1].h.frozen = True

    def __call__(self, x, y):
        for v, l in zip(self.vodes[:-1], self.layers[:-1]):
            x = v(self.act_fn(l(x)))

        x = self.vodes[-1](self.layers[-1](x))

        if y is not None:
            self.vodes[-1].set("h", y)

        return self.vodes[-1].get("u")


# ==============================================================================
# Transform Functions
# ==============================================================================

@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: Model):
    return model(x, y)


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: Model):
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
    model: Model,
    optim_w: pxu.Optim,
    optim_h: pxu.Optim
):
    model.train()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)

    for i in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, y_), g = pxf.value_and_grad(
                pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
                has_aux=True
            )(energy)(x, model=model)
        
        optim_h.step(model, g["model"])

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, y_), g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)(x, model=model)
    optim_w.step(model, g["model"], scale_by=1.0/x.shape[0])


@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: Model):
    model.eval()
    
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model).argmax(axis=-1)
    
    return (y_ == y).mean(), y_


def train(dl, T, *, model: Model, optim_w: pxu.Optim, optim_h: pxu.Optim):
    for x, y in dl:
        train_on_batch(T, x, jax.nn.one_hot(y, 2), model=model, optim_w=optim_w, optim_h=optim_h)


def eval(dl, *, model: Model):
    acc = []
    ys_ = []
    
    for x, y in dl:
        a, y_ = eval_on_batch(x, y, model=model)
        acc.append(a)
        ys_.append(y_)
    
    return np.mean(acc), np.concatenate(ys_)


# ==============================================================================
# Initialization Helper
# ==============================================================================

def init(
    batch_size: int,
    hidden_dim: int,
    h_lr: float = 1e-2,
    nm_layers: int = 4,
    act_fn: Callable[[jax.Array], jax.Array] = jax.nn.leaky_relu
):
    """Initialize model and optimizers."""
    model = Model(
        input_dim=2,
        hidden_dim=hidden_dim,
        output_dim=2,
        nm_layers=nm_layers,
        act_fn=act_fn
    )
    
    # Perform dummy forward pass to initialize Vodes
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jax.numpy.zeros((batch_size, 2)), None, model=model)
        
        optim_h = pxu.Optim(lambda: optax.sgd(h_lr), pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
        optim_w = pxu.Optim(lambda: optax.adamw(1e-2), pxu.M(pxnn.LayerParam)(model))
    
    return model, optim_h, optim_w


# ==============================================================================
# Experiment Runner
# ==============================================================================

def run(
    nm_epochs,
    model,
    optim_h,
    optim_w,
    train_dl,
    test_dl,
    T=8
):
    """Run training for specified number of epochs."""
    for _ in range(nm_epochs):
        random.shuffle(train_dl)
        train(train_dl, T=T, model=model, optim_w=optim_w, optim_h=optim_h)
    a, _ = eval(test_dl, model=model)
    
    return a


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    print("=" * 80)
    print("Tutorial #1: Analysis of PCNs")
    print("=" * 80)
    print("\nThis analysis studies how hidden width and h learning rate affect PCN training.")
    print("It explores the relationship between network architecture and optimal hyperparameters.")
    
    # Setup
    batch_size = 64
    nm_layers = 4
    nm_elements = 512
    nm_epochs = 32 // (nm_elements // batch_size)
    
    print(f"\nExperiment configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of layers: {nm_layers}")
    print(f"  Number of epochs: {nm_epochs}")
    print(f"  Inference steps (T): 8")
    
    # Generate dataset
    X, y = make_moons(n_samples=batch_size * (nm_elements // batch_size), noise=0.2, random_state=42)
    train_dl = list(zip(X.reshape(-1, batch_size, 2), y.reshape(-1, batch_size)))
    
    X_test, y_test = make_moons(n_samples=batch_size * (nm_elements // batch_size) // 2, noise=0.2, random_state=0)
    test_dl = tuple(zip(X_test.reshape(-1, batch_size, 2), y_test.reshape(-1, batch_size)))
    
    print(f"\nDataset: {X.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Test values
    h_dims = [16, 32, 64, 128, 256, 512, 1024, 2048]
    h_lrs = [1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 5e-1]
    
    print(f"\nTesting {len(h_dims)} hidden dimensions: {h_dims}")
    print(f"Testing {len(h_lrs)} learning rates: {h_lrs}")
    print(f"\nTotal experiments: {len(h_dims) * len(h_lrs)}")
    
    # Run experiments
    print("\n" + "=" * 80)
    print("Starting experiments...")
    print("=" * 80)
    
    acc_by_dim = {}
    
    for h_dim in h_dims:
        acc_by_dim[h_dim] = []
        print(f"\n--- Hidden dimension: {h_dim} ---")
        
        for h_lr in h_lrs:
            print(f"  h_lr={h_lr:.4f}...", end=" ", flush=True)
            model, optim_h, optim_w = init(batch_size, h_dim, h_lr, act_fn=jax.nn.leaky_relu)
            a = run(nm_epochs, model, optim_h, optim_w, train_dl, test_dl, 8)
            acc_by_dim[h_dim].append(a)
            print(f"Accuracy: {(a*100.0):.2f}%")
    
    # Summary
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print("\nBest accuracy for each hidden dimension:")
    print(f"{'Hidden Dim':<12} {'Best Acc':<10} {'Best h_lr':<10}")
    print("-" * 32)
    
    for h_dim in h_dims:
        accs = acc_by_dim[h_dim]
        max_acc = max(accs)
        best_lr_idx = accs.index(max_acc)
        best_lr = h_lrs[best_lr_idx]
        print(f"{h_dim:<12} {max_acc*100:.2f}%      {best_lr:<10.4f}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print("\nKey finding: Best learning rate changes with hidden dimension size.")
    print("High hidden dims prefer small learning rates and are unstable for larger values.")
    print("This differs from backpropagation, which is unaffected by network width.")


if __name__ == "__main__":
    main()