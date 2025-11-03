"""
Tutorial #0: Predictive Coding Networks (PCNs)

This script demonstrates how to create and train a simple PCN to classify the two moons dataset.
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

# Set random seed for reproducibility
px.RKG.seed(0)


# ==============================================================================
# Model Definition
# ==============================================================================

class Model(pxc.EnergyModule):
    """
    PCN model for classification.
    
    The model consists of layers and vodes (vectorized nodes). Each vode maintains
    a hidden state and computes energy based on prediction errors.
    """
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
        
        # Define layers
        self.layers = [pxnn.Linear(input_dim, hidden_dim)] + [
            pxnn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 2)
        ] + [pxnn.Linear(hidden_dim, output_dim)]

        # Define vodes (vectorized nodes)
        # The default ruleset for a Vode is: {"STATUS.INIT": ("h, u <- u",),}
        # This means: if status is STATUS.INIT, when setting 'u', also save it to 'h'
        # This implements forward initialization
        self.vodes = [
            pxc.Vode() for _ in range(nm_layers - 1)
        ] + [pxc.Vode(pxc.ce_energy)]  # Use cross-entropy for classification
        
        # Freeze the last vode's hidden state (it will hold the target)
        self.vodes[-1].h.frozen = True

    def __call__(self, x, y):
        for v, l in zip(self.vodes[:-1], self.layers[:-1]):
            # Forward pass: x = activation(layer(x)), then store in vode
            x = v(self.act_fn(l(x)))

        x = self.vodes[-1](self.layers[-1](x))

        if y is not None:
            # Set target for training
            self.vodes[-1].set("h", y)

        return self.vodes[-1].get("u")


# ==============================================================================
# Transform Functions
# ==============================================================================

@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: Model):
    """Forward pass through the model."""
    return model(x, y)


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: Model):
    """Compute total energy of the model."""
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
    """Train on a single batch."""
    print("Compiling train_on_batch...")  # This prints only during compilation
    
    model.train()
    
    # Init step: forward pass to initialize vodes
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)
    
    # Initialize optimizer for current batch
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    
    # Inference steps: update hidden states to minimize energy
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, y_), g = pxf.value_and_grad(
                pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
                has_aux=True
            )(energy)(x, model=model)
        
        optim_h.step(model, g["model"])
    
    optim_h.clear()

    # Weight update step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, y_), g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)(x, model=model)
    
    # Scale gradient by batch size
    optim_w.step(model, g["model"], scale_by=1.0/x.shape[0])


@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: Model):
    """Evaluate on a single batch."""
    model.eval()
    
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model).argmax(axis=-1)
    
    return (y_ == y).mean(), y_


def train(dl, T, *, model: Model, optim_w: pxu.Optim, optim_h: pxu.Optim):
    """Standard training loop."""
    for x, y in dl:
        train_on_batch(T, x, jax.nn.one_hot(y, 2), model=model, optim_w=optim_w, optim_h=optim_h)


def eval(dl, *, model: Model):
    """Standard evaluation loop."""
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
    print("Tutorial #0: Predictive Coding Networks (PCNs)")
    print("=" * 80)
    
    # Hyperparameters
    batch_size = 32
    nm_elements = 1024
    nm_epochs = 256 // (nm_elements // batch_size)
    
    print(f"\nHyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of elements: {nm_elements}")
    print(f"  Number of epochs: {nm_epochs}")
    
    # Create model
    model = Model(
        input_dim=2,
        hidden_dim=32,
        output_dim=2,
        nm_layers=3,
        act_fn=jax.nn.leaky_relu
    )
    print(f"\nModel created with {len(model.layers)} layers")
    
    # Create optimizers
    import optax
    optim_w = pxu.Optim(lambda: optax.adamw(1e-2), pxu.M(pxnn.LayerParam)(model))
    optim_h = pxu.Optim(lambda: optax.sgd(1e-2, momentum=0.5, nesterov=True))
    print("Optimizers initialized")
    
    # Generate dataset
    X, y = make_moons(n_samples=batch_size * (nm_elements // batch_size), noise=0.2, random_state=42)
    print(f"\nDataset generated: {X.shape[0]} samples")
    
    # Split into batches
    train_dl = list(zip(X.reshape(-1, batch_size, 2), y.reshape(-1, batch_size)))
    
    X_test, y_test = make_moons(n_samples=batch_size * (nm_elements // batch_size) // 2, noise=0.2, random_state=0)
    test_dl = tuple(zip(X_test.reshape(-1, batch_size, 2), y_test.reshape(-1, batch_size)))
    
    print(f"Training batches: {len(train_dl)}")
    print(f"Test batches: {len(test_dl)}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    T = 8  # Number of inference steps
    
    for e in range(nm_epochs):
        random.shuffle(train_dl)
        train(train_dl, T=T, model=model, optim_w=optim_w, optim_h=optim_h)
        a, y_pred = eval(test_dl, model=model)
        
        print(f"Epoch {e + 1}/{nm_epochs} - Test Accuracy: {a * 100:.2f}%")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # Note about recompilation
    print("\nNote: 'Compiling train_on_batch...' appears only during first compilation.")
    print("If it appears twice, it's due to Vode state initialization on first batch.")


if __name__ == "__main__":
    main()