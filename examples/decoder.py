"""
Tutorial #5: Decoder-only PCN

This script demonstrates how to code a decoder-only PCN and train it to generate
FashionMNIST images. It introduces the concept of rulesets for customizing Vode behavior.
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


# Custom status for forward mode
STATUS_FORWARD = "forward"


# ==============================================================================
# Model Definition
# ==============================================================================

class Decoder(pxc.EnergyModule):
    """
    Decoder-only PCN for generative modeling.
    
    This model uses custom rulesets to:
    - Initialize the first node to zeros (no input)
    - Enable a "forward mode" where we forward activations instead of node states
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nm_layers: int,
        act_fn: Callable[[jax.Array], jax.Array],
    ) -> None:
        super().__init__()

        self.act_fn = px.static(act_fn)

        self.layers = (
            [pxnn.Linear(input_dim, hidden_dim)]
            + [pxnn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 2)]
            + [pxnn.Linear(hidden_dim, output_dim)]
        )

        # First vode: initialize to zero (no input)
        # Custom ruleset: when STATUS.INIT, set h and u to zeros
        self.vodes = [
            pxc.Vode(
                energy_fn=None,  # No prior on first layer
                ruleset={pxc.STATUS.INIT: ("h, u <- u:to_zero",)},
                tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros((input_dim,))},
            )
        ] + [
            # Remaining vodes: 
            # - Initialize to zeros
            # - Enable forward mode where h -> u (forward activation instead of state)
            pxc.Vode(
                ruleset={
                    pxc.STATUS.INIT: ("h, u <- u:to_zero",),
                    STATUS_FORWARD: ("h -> u",)
                },
                tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros_like(v)},
            )
            for _ in range(nm_layers - 1)
        ] + [pxc.Vode()]
        
        self.vodes[-1].h.frozen = True

    def __call__(self, y: jax.Array | None):
        # First vode always returns its state (starts from zero)
        x = self.vodes[0](jnp.empty(()))
        
        for i, layer in enumerate(self.layers):
            # Apply activation except on last layer
            act_fn = self.act_fn if i != len(self.layers) - 1 else lambda x: x
            x = act_fn(layer(x))
            x = self.vodes[i + 1](x)

        if y is not None:
            self.vodes[-1].set("h", y.flatten())

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
    """Custom DataLoader that returns numpy arrays."""
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
    """Create FashionMNIST train and test dataloaders."""
    t = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.FashionMNIST(
        "~/tmp/fashion-mnist/",
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
        "~/tmp/fashion-mnist/",
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

@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=0, out_axes=0)
def forward(x, *, model: Decoder):
    return model(x)


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), out_axes=(None, 0), axis_name="batch")
def energy(*, model: Decoder):
    y_ = model(None)
    return jax.lax.psum(model.energy(), "batch"), y_


# ==============================================================================
# Training Functions
# ==============================================================================

@pxf.jit(static_argnums=0)
def train_on_batch(T: int, x: jax.Array, *, model: Decoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
    model.train()

    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(
        energy
    )

    learning_step = pxf.value_and_grad(pxu.M_hasnot(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, model=model)

    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = inference_step(model=model)

        optim_h.step(model, g["model"])
    
    optim_h.clear()

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = learning_step(model=model)
    optim_w.step(model, g["model"], scale_by=1.0/x.shape[0])


@pxf.jit(static_argnums=0)
def eval_on_batch(T: int, x: jax.Array, *, model: Decoder, optim_h: pxu.Optim):
    """Evaluate by inferring hidden states and computing reconstruction loss."""
    model.train()

    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(
        energy
    )

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, model=model)
    
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = inference_step(model=model)

        optim_h.step(model, g["model"])
    
    optim_h.clear()

    # Generate output using forward mode
    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        x_hat = forward(None, model=model)

    # Compute reconstruction loss
    loss = jnp.square(jnp.clip(x_hat.flatten(), 0.0, 1.0) - x.flatten()).mean()

    return loss, x_hat


def train(dl, T, *, model: Decoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
    for i, (x, y) in enumerate(dl):
        train_on_batch(T, x, model=model, optim_w=optim_w, optim_h=optim_h)
        
        if i % 100 == 0:
            print(f"  Batch {i}/{len(dl)}", flush=True)


def eval(dl, T, *, model: Decoder, optim_h: pxu.Optim):
    losses = []

    for x, y in dl:
        e, y_hat = eval_on_batch(T, x, model=model, optim_h=optim_h)
        losses.append(e)

    return np.mean(losses)


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    print("=" * 80)
    print("Tutorial #5: Decoder-only PCN")
    print("=" * 80)
    print("\nThis tutorial demonstrates generative modeling with PCNs.")
    print("We train a decoder to generate FashionMNIST images.\n")
    
    # Hyperparameters
    batch_size = 128
    nm_epochs = 24
    T = 20  # Inference steps
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {nm_epochs}")
    print(f"  Inference steps (T): {T}")
    
    # Create model
    model = Decoder(
        input_dim=64,
        hidden_dim=256,
        output_dim=28 * 28,
        nm_layers=4,
        act_fn=jax.nn.tanh
    )
    print(f"\nModel created:")
    print(f"  Input dim: 64 (latent space)")
    print(f"  Hidden dim: 256")
    print(f"  Output dim: 784 (28x28 images)")
    print(f"  Layers: 4")
    
    # Create optimizers
    optim_h = pxu.Optim(lambda: optax.sgd(5e-2, momentum=0.1))
    optim_w = pxu.Optim(lambda: optax.adamw(1e-4), pxu.M(pxnn.LayerParam)(model))
    print("Optimizers initialized")
    
    # Load data
    print("\nLoading FashionMNIST dataset...")
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
        l = eval(test_dataloader, T=T, model=model, optim_h=optim_h)
        print(f"  Test Loss: {l:.4f}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    print("\nKey concepts demonstrated:")
    print("  - Custom rulesets for Vode behavior")
    print("  - Zero initialization for decoder input")
    print("  - Forward mode for generation")
    print("  - Generative modeling with PCNs")


if __name__ == "__main__":
    main()