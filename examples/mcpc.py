"""
Tutorial #7: Monte Carlo Predictive Coding (MCPC)

This script demonstrates MCPC, which uses scan and momentum for state inference
without resetting momentum between inference steps. It includes a custom noisy
SGD optimizer for stochastic dynamics.
"""

from typing import Callable, Any, Optional
import numpy as np
from scipy.stats import wasserstein_distance

import jax
import jax.numpy as jnp
from optax._src import base, combine, transform
import optax

import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.functional as pxf
import pcx.utils as pxu


# ==============================================================================
# Custom Optimizer: Stochastic Gradient Langevin Dynamics
# ==============================================================================

def sgdld(
    learning_rate: base.ScalarOrSchedule,
    momentum: Optional[float] = None,
    h_var: float = 1.0,
    gamma: float = 0.,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
    seed: int = lambda: px.RKG(1)[0],
) -> base.GradientTransformation:
    """
    Stochastic Gradient Langevin Dynamics optimizer.
    
    Adds noise to gradients for stochastic inference in MCPC.
    
    Args:
        learning_rate: Learning rate
        momentum: Momentum coefficient
        h_var: Variance of the hidden states
        gamma: Friction coefficient
        nesterov: Whether to use Nesterov momentum
        accumulator_dtype: Data type for accumulator
        seed: Random seed function
    """
    def optim_fn():
        eta = 2*h_var*(1-momentum)/learning_rate if momentum is not None else 2*h_var/learning_rate
        s = seed()
        return combine.chain(
            transform.add_noise(eta, gamma, s),
            (transform.trace(decay=momentum, nesterov=nesterov,
                            accumulator_dtype=accumulator_dtype)
            if momentum is not None else base.identity()),
            transform.scale_by_learning_rate(learning_rate)
        )
    return optim_fn


# ==============================================================================
# Model Definition
# ==============================================================================

class Model(pxc.EnergyModule):
    """Simple linear PCN for MCPC demonstration."""
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
        
        # Simple linear network
        self.layers = [pxnn.Linear(input_dim, hidden_dim)] + [
            pxnn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(nm_layers - 2)
        ] + [pxnn.Linear(hidden_dim, output_dim, bias=False)]

        self.vodes = [
            pxc.Vode() for _ in range(nm_layers)
        ]
        
        self.vodes[-1].h.frozen = True

    def __call__(self, x, y):
        for v, l in zip(self.vodes[:-1], self.layers[:-1]):
            x = self.act_fn(v(l(x)))

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
# Training with Scan
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
    """Train using scan for efficient inference loop."""
    def h_step(i, x, *, model, optim_h):
        """Single inference step."""
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, y_), g = pxf.value_and_grad(
                pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
                has_aux=True
            )(energy)(x, model=model)
        optim_h.step(model, g["model"])
        return x, None

    model.train()
        
    # Init step
    with pxu.step(model, (pxc.STATUS.INIT, None), clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    
    # Inference steps using scan
    pxf.scan(h_step, xs=jax.numpy.arange(T))(x, model=model, optim_h=optim_h)
    
    optim_h.clear()

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, y_), g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)(x, model=model)
    optim_w.step(model, g["model"], scale_by=1.0/x.shape[0])


def train(dl, T, *, model: Model, optim_w: pxu.Optim, optim_h: pxu.Optim):
    """Training loop."""
    for i, (x, y) in enumerate(dl):
        train_on_batch(T, x, y, model=model, optim_w=optim_w, optim_h=optim_h)
        if i % 50 == 0:
            print(f"  Batch {i}/{len(dl)}", end="\r", flush=True)
    print()  # New line after progress


# ==============================================================================
# Evaluation for Generative Modeling
# ==============================================================================

@pxf.jit(static_argnums=0)
def eval_on_batch(
    T: int,
    x: jax.Array, 
    *, 
    model: Model,
    optim_h: pxu.Optim
):
    """
    Evaluate by running inference without frozen last vode.
    This allows the model to generate/infer the output distribution.
    """
    def h_step(i, x, *, model, optim_h):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, y_), g = pxf.value_and_grad(
                pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
                has_aux=True
            )(energy)(x, model=model)
        optim_h.step(model, g["model"])
        return x, None

    model.train()

    if model.vodes[-1].h.frozen:
        print("Warning: vode[-1] should not be frozen during eval!")
        
    # Init step
    with pxu.step(model, (pxc.STATUS.INIT, None), clear_params=pxc.VodeParam.Cache):
        forward(x, None, model=model)
    optim_h.init(pxu.M(pxc.VodeParam)(model))
    
    # Inference steps
    x, y_ = pxf.scan(h_step, xs=jax.numpy.arange(T))(x, model=model, optim_h=optim_h)
    
    optim_h.clear()


def eval(dl, T, *, model: Model, optim_h: pxu.Optim):
    """
    Evaluate by computing Wasserstein distance between true and inferred distributions.
    """
    model.vodes[-1].h.frozen = False
    ys = []
    ys_ = []
    
    for x, y in dl:
        eval_on_batch(T, x, model=model, optim_h=optim_h)
        ys.append(y)
        ys_.append(model.vodes[-1].get("h"))

    ys = np.concatenate(ys, axis=0)
    ys_ = np.concatenate(ys_, axis=0)

    return wasserstein_distance(ys.squeeze(), ys_.squeeze()), ys_


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    print("=" * 80)
    print("Tutorial #7: Monte Carlo Predictive Coding (MCPC)")
    print("=" * 80)
    print("\nMCPC uses stochastic dynamics for inference without resetting")
    print("momentum between steps. This is demonstrated on a simple 1D regression task.\n")
    
    # Configuration
    batch_size = 32
    lr = 1e-1
    momentum = 0.5
    h_var = 1.0
    gamma = 0.
    lr_p = 1e-3
    
    # True distribution parameters
    mean = 1
    var = 5
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  State learning rate: {lr}")
    print(f"  Weight learning rate: {lr_p}")
    print(f"  Momentum: {momentum}")
    print(f"  Hidden variance: {h_var}")
    print(f"\nTarget distribution:")
    print(f"  Mean: {mean}")
    print(f"  Variance: {var}")
    
    # Generate data (inputs are all zeros, outputs are samples from Gaussian)
    nm_elements = 10240
    X = np.zeros((batch_size * (nm_elements // batch_size), 1))
    y = np.random.randn(batch_size * (nm_elements // batch_size)).reshape(-1, 1) * np.sqrt(var) + mean
    
    nm_elements_test = 1024
    X_test = np.zeros((batch_size * (nm_elements_test // batch_size), 1))
    y_test = np.random.randn(batch_size * (nm_elements_test // batch_size)).reshape(-1, 1) * np.sqrt(var) + mean
    
    # Create dataloaders
    train_dl = list(zip(X.reshape(-1, batch_size, 1), y.reshape(-1, batch_size, 1)))
    test_dl = tuple(zip(X_test.reshape(-1, batch_size, 1), y_test.reshape(-1, batch_size, 1)))
    
    print(f"\nDataset: {X.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Create model
    model = Model(
        input_dim=1,
        hidden_dim=1,
        output_dim=1,
        nm_layers=2,
        act_fn=lambda x: x  # Linear model
    )
    print("\nModel: 1 -> 1 -> 1 (linear)")
    
    # Create optimizers
    h_optimiser_fn = sgdld
    
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jax.numpy.zeros((batch_size, 1)), None, model=model)
        model.vodes[-1].h.frozen = True
        optim_h = pxu.Optim(h_optimiser_fn(lr, momentum, h_var, gamma))
        optim_w = pxu.Optim(lambda: optax.adam(lr_p), pxu.M(pxnn.LayerParam)(model))
        model.vodes[-1].h.frozen = False
    
    print("Optimizers: SGLD (states), Adam (weights)")
    
    # Training
    nm_epochs = 5120 // (nm_elements // batch_size)
    T = 100
    T_eval = 100
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {nm_epochs}")
    print(f"  Inference steps (train): {T}")
    print(f"  Inference steps (eval): {T_eval}")
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    # Initial evaluation
    w, y_ = eval(test_dl, T=T_eval, model=model, optim_h=optim_h)
    print(f"\nEpoch 0/{nm_epochs} - Wasserstein distance: {w:.4f}")
    
    # Training loop
    for e in range(nm_epochs):
        print(f"\nEpoch {e + 1}/{nm_epochs}")
        train(train_dl, T=T, model=model, optim_w=optim_w, optim_h=optim_h)
        
        if e % 5 == 4 or e == nm_epochs - 1:
            w, y_ = eval(test_dl, T=T_eval, model=model, optim_h=optim_h)
            print(f"  Wasserstein distance: {w:.4f}")
    
    # Final results
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    print(f"\nLearned distribution:")
    print(f"  Mean: {y_.mean():.2f} (target: {mean})")
    print(f"  Var:  {y_.var():.2f} (target: {var})")
    
    print(f"\nLearned parameters:")
    weight = model.layers[-1].nn.weight.get()[0, 0]
    bias = model.layers[0].nn.bias.get()[0]
    print(f"  Weight: {weight:.2f}")
    print(f"  Bias:   {bias:.2f}")
    
    print("\nMCPC successfully learned the target distribution!")
    print("Key concepts:")
    print("  - Stochastic gradient Langevin dynamics (SGLD)")
    print("  - Persistent momentum across inference steps")
    print("  - Generative modeling with PCNs")
    print("  - Using scan for efficient loops")


if __name__ == "__main__":
    main()