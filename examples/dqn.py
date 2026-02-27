"""
Deep Equilibrium Predictive Coding (DEQ-PC) on MNIST - BALANCED VERSION
========================================================================
Key insight: Scale down classification energy to let DEQ dynamics dominate.
Total energy = E_z + E_vmid + β * E_vout, where β << 1
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.datasets import fetch_openml
from typing import Callable

import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.functional as pxf
import pcx.utils as pxu


# =============================================================================
# Model Definition
# =============================================================================

class DEQ_PC(pxc.EnergyModule):
    """
    DEQ-PC with balanced energy:
    E_total = E_z + E_vmid + β * E_vout
    
    β << 1 ensures fixed-point dynamics dominate, with classification as gentle guidance.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 act_fn: Callable, alpha: float = 0.5, beta: float = 0.1):
        super().__init__()
        
        self.act_fn = px.static(act_fn)
        self.hidden_dim = px.static(hidden_dim)
        self.alpha = px.static(alpha)  # Residual connection
        self.beta = px.static(beta)    # Classification energy weight
        
        self.embed = pxnn.Linear(input_dim, hidden_dim)
        self.layer1 = pxnn.Linear(hidden_dim, hidden_dim)
        self.layer2 = pxnn.Linear(hidden_dim, hidden_dim)
        self.readout = pxnn.Linear(hidden_dim, output_dim)
        
        self.z = pxc.Vode(energy_fn=pxc.se_energy)
        self.v_mid = pxc.Vode(energy_fn=pxc.se_energy)
        self.v_out = pxc.Vode(energy_fn=pxc.ce_energy)
        self.v_out.h.frozen = True
    
    def __call__(self, x, y=None):
        z_h = self.z.get("h")
        alpha = self.alpha.get()
        
        # f(z, x) = layer2(act(layer1(z + embed(x))))
        combined = z_h + self.embed(x)
        mid_activation = self.act_fn(self.layer1(combined))
        self.v_mid(mid_activation)
        
        f_z = self.layer2(self.v_mid.get("h"))
        
        # Contractive: z.u = α*z.h + (1-α)*f(z.h, x)
        z_target = alpha * z_h + (1 - alpha) * f_z
        self.z(z_target)
        
        # Classification
        self.v_out(self.readout(z_h))
        if y is not None:
            self.v_out.set("h", y)
        
        return self.v_out.get("u")
    
    def energy(self):
        """Balanced energy with scaled classification term."""
        beta = self.beta.get()
        return self.z.energy() + self.v_mid.energy() + beta * self.v_out.energy()
    
    def energy_deq_only(self):
        """DEQ energy only (for eval)."""
        return self.z.energy() + self.v_mid.energy()


# =============================================================================
# Weight Clipping
# =============================================================================

def clip_weights(model, max_norm: float = 1.0):
    for layer in [model.layer1, model.layer2]:
        w = layer.nn.weight.get()
        norm = jnp.linalg.norm(w)
        scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
        layer.nn.weight.set(w * scale)


# =============================================================================
# Batched Operations
# =============================================================================

@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model):
    return model(x, y)


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy_train(x, *, model):
    y_pred = model(x, None)
    e_total = model.energy()  # Uses balanced energy
    return jax.lax.psum(e_total, "batch"), y_pred


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy_deq(x, *, model):
    y_pred = model(x, None)
    e_deq = model.energy_deq_only()
    return jax.lax.psum(e_deq, "batch"), y_pred


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=(None, None, None, 0), axis_name="batch")
def energy_components(x, *, model):
    y_pred = model(x, None)
    e_z = jax.lax.psum(model.z.energy(), "batch")
    e_vmid = jax.lax.psum(model.v_mid.energy(), "batch")
    e_vout = jax.lax.psum(model.v_out.energy(), "batch")
    return e_z, e_vmid, e_vout, y_pred


# =============================================================================
# Training Step
# =============================================================================

@pxf.jit(static_argnums=0)
def train_step(T: int, x: jax.Array, y: jax.Array, *, model, optim_h, optim_w):
    model.train()
    batch_size = x.shape[0]
    hidden_dim = model.hidden_dim.get()
    
    z_init = jnp.zeros((batch_size, hidden_dim))
    model.z.set("h", z_init)
    
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)
    model.z.set("h", z_init)
    
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, _), g = pxf.value_and_grad(
                pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
                has_aux=True
            )(energy_train)(x, model=model)
        optim_h.step(model, g["model"])
    
    optim_h.clear()
    
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, _), g = pxf.value_and_grad(
            pxu.M(pxnn.LayerParam).to([False, True]),
            has_aux=True
        )(energy_train)(x, model=model)
    optim_w.step(model, g["model"], scale_by=1.0 / batch_size)
    
    return e


# =============================================================================
# Evaluation Step
# =============================================================================

@pxf.jit(static_argnums=0)
def eval_step(T: int, x: jax.Array, y: jax.Array, *, model, optim_h):
    model.train()
    batch_size = x.shape[0]
    hidden_dim = model.hidden_dim.get()
    
    z_init = jnp.zeros((batch_size, hidden_dim))
    model.z.set("h", z_init)
    
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, None, model=model)
    model.z.set("h", z_init)
    
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    
    # Only minimize DEQ energy during eval (no label available)
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, _), g = pxf.value_and_grad(
                pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
                has_aux=True
            )(energy_deq)(x, model=model)
        optim_h.step(model, g["model"])
    
    optim_h.clear()
    
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        y_pred = forward(x, None, model=model).argmax(axis=-1)
    
    return (y_pred == y).sum()


# =============================================================================
# Debug Training Step
# =============================================================================

@pxf.jit(static_argnums=0)
def train_step_debug(T: int, x: jax.Array, y: jax.Array, *, model, optim_h, optim_w):
    model.train()
    batch_size = x.shape[0]
    hidden_dim = model.hidden_dim.get()
    
    z_init = jnp.zeros((batch_size, hidden_dim))
    model.z.set("h", z_init)
    
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)
    model.z.set("h", z_init)
    
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    
    # First iteration
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e_first, _), g = pxf.value_and_grad(
            pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
            has_aux=True
        )(energy_train)(x, model=model)
    optim_h.step(model, g["model"])
    
    for _ in range(T - 1):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, _), g = pxf.value_and_grad(
                pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
                has_aux=True
            )(energy_train)(x, model=model)
        optim_h.step(model, g["model"])
    
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e_last, _), _ = pxf.value_and_grad(
            pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
            has_aux=True
        )(energy_train)(x, model=model)
        e_z, e_vmid, e_vout, _ = energy_components(x, model=model)
    
    optim_h.clear()
    
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)
        z_h = model.z.get("h")
        z_u = model.z.get("u")
    fp_residual = jnp.linalg.norm(z_h - z_u) / batch_size
    
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, _), g = pxf.value_and_grad(
            pxu.M(pxnn.LayerParam).to([False, True]),
            has_aux=True
        )(energy_train)(x, model=model)
    optim_w.step(model, g["model"], scale_by=1.0 / batch_size)
    
    return e_first, e_last, e_z, e_vmid, e_vout, fp_residual


# =============================================================================
# Main
# =============================================================================

def main():
    px.RKG.seed(42)
    
    print("Loading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)
    
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]
    
    batch_size = 256
    n_train = (len(X_train) // batch_size) * batch_size
    n_test = (len(X_test) // batch_size) * batch_size
    
    X_train_batches = X_train[:n_train].reshape(-1, batch_size, 784)
    y_train_batches = y_train[:n_train].reshape(-1, batch_size)
    X_test_batches = X_test[:n_test].reshape(-1, batch_size, 784)
    y_test_batches = y_test[:n_test].reshape(-1, batch_size)
    
    print(f"Train: {len(X_train_batches)} batches, Test: {len(X_test_batches)} batches")
    
    # Hyperparameters
    hidden_dim = 256      # Increased from 32
    alpha = 0.5           # Residual connection strength  
    beta = 0.1          # Classification energy weight (small!)
    T = 500                # More inference steps
    n_epochs = 20
    lr_h = 0.05            # State learning rate
    lr_w = 1e-4           # Weight learning rate
    
    print(f"\nHyperparameters:")
    print(f"  hidden_dim = {hidden_dim}")
    print(f"  alpha (residual) = {alpha}")
    print(f"  beta (class weight) = {beta}")
    print(f"  T (inference steps) = {T}")
    print(f"  lr_h = {lr_h}, lr_w = {lr_w}")
    
    # Initialize model
    model = DEQ_PC(
        input_dim=784, 
        hidden_dim=hidden_dim, 
        output_dim=10, 
        act_fn=jax.nn.tanh,
        alpha=alpha,
        beta=beta
    )
    
    # Warm up
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        model.z.set("h", jnp.zeros((batch_size, hidden_dim)))
        forward(jnp.zeros((batch_size, 784)), None, model=model)
    
    # Optimizers
    optim_h = pxu.Optim(lambda: optax.sgd(lr_h, momentum=0.9))
    optim_w = pxu.Optim(
        lambda: optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(lr_w, weight_decay=1e-5)
        ), 
        pxu.M(pxnn.LayerParam)(model)
    )
    
    print("\n" + "=" * 95)
    print(f"Training DEQ-PC (β={beta}, hidden={hidden_dim})")
    print("=" * 95)
    print(f"{'Epoch':>5} {'Acc':>7} {'E_start':>10} {'E_end':>10} {'E_z':>9} {'E_vmid':>9} {'βE_vout':>9} {'FP_res':>8}")
    print("-" * 95)
    
    best_acc = 0
    
    for epoch in range(n_epochs):
        # Debug first batch
        e_first, e_last, e_z, e_vmid, e_vout, fp_res = train_step_debug(
            T, X_train_batches[0], jax.nn.one_hot(y_train_batches[0], 10),
            model=model, optim_h=optim_h, optim_w=optim_w
        )
        
        clip_weights(model, max_norm=3.0)
        
        # Train remaining batches
        for i in range(1, len(X_train_batches)):
            train_step(
                T, X_train_batches[i], jax.nn.one_hot(y_train_batches[i], 10),
                model=model, optim_h=optim_h, optim_w=optim_w
            )
            if i % 50 == 0:
                clip_weights(model, max_norm=3.0)
        
        clip_weights(model, max_norm=3.0)
        
        # Evaluate
        correct = sum(
            eval_step(T, X_test_batches[i], y_test_batches[i], model=model, optim_h=optim_h)
            for i in range(len(X_test_batches))
        )
        acc = correct / n_test * 100
        best_acc = max(best_acc, acc)
        
        # Note: βE_vout is what actually contributes to total energy
        beta_e_vout = beta * float(e_vout)
        
        print(f"{epoch+1:>5} {acc:>6.2f}% {float(e_first):>10.1f} {float(e_last):>10.1f} "
              f"{float(e_z):>9.2f} {float(e_vmid):>9.2f} {beta_e_vout:>9.2f} {float(fp_res):>8.4f}")
    
    print("=" * 95)
    print(f"\nBest accuracy: {best_acc:.2f}%")
    
    # Diagnostics
    print("\nFinal diagnostics:")
    print(f"  E_z (fixed-point): {float(e_z):.2f}")
    print(f"  E_vmid (intermediate): {float(e_vmid):.2f}")
    print(f"  E_vout (classification, unscaled): {float(e_vout):.2f}")
    print(f"  β*E_vout (classification, scaled): {beta * float(e_vout):.2f}")
    print(f"  Fixed-point residual: {float(fp_res):.6f}")
    
    w1_norm = float(jnp.linalg.norm(model.layer1.nn.weight.get()))
    w2_norm = float(jnp.linalg.norm(model.layer2.nn.weight.get()))
    print(f"  Weight norms: layer1={w1_norm:.3f}, layer2={w2_norm:.3f}")


if __name__ == "__main__":
    main()