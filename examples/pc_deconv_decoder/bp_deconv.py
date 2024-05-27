from typing import Callable
from pathlib import Path

# Core dependencies
import jax
import jax.numpy as jnp
import numpy as np
import optax

# pcax
import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.utils as pxu
import pcax.functional as pxf

from conv_transpose_layer import ConvTranspose
from data_utils import load_cifar10, get_batches, reconstruct_image


STATUS_FORWARD = "forward"


class BPDeconvDecoder(pxc.EnergyModule):
    def __init__(
        self,
        input_dim: tuple[int, int, int] = (8, 8, 8),
        bottleneck_dim: int = 128,
        *,
        act_fn: Callable[[jax.Array], jax.Array],
        output_act_fn: Callable[[jax.Array], jax.Array] = lambda x: x,
    ):
        super().__init__()
        self.act_fn = px.static(act_fn)
        self.output_act_fn = px.static(output_act_fn)
        self.input_dim = px.static(input_dim)

        self.layers = [
            pxnn.Conv2d(3, 5, kernel_size=(5, 5), stride=(2, 2), padding=2),  # (16, 16, 5)
            self.act_fn,
            pxnn.Conv2d(5, 8, kernel_size=(5, 5), stride=(2, 2), padding=2),  # (8, 8, 8)
            self.act_fn,
            pxnn.Conv2d(8, 8, kernel_size=(5, 5), stride=(2, 2), padding=2),  # (4, 4, 8)
            self.act_fn,
            ConvTranspose(2, 8, 8, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),  # (8, 8, 8)
            self.act_fn,
            ConvTranspose(2, 8, 5, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),  # (16, 16, 5)
            self.act_fn,
            ConvTranspose(2, 5, 3, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),  # (32, 32, 3)
            self.output_act_fn,
        ]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=0, out_axes=0)
def forward(x: jax.Array | None = None, *, model: BPDeconvDecoder) -> jax.Array:
    return model(x)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=0, out_axes=(None, 0), axis_name="batch")
def loss(y: jax.Array, *, model: BPDeconvDecoder) -> jax.Array:
    y_ = model(y)
    return jax.lax.pmean(0.5 * jnp.sum((y - y_) ** 2), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch(T: int, examples: jax.Array, *, model: BPDeconvDecoder, optim_w: pxu.Optim):
    model.train()

    learning_step = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(loss)

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = learning_step(examples, model=model)
    optim_w.step(model, g["model"])


@pxf.jit(static_argnums=0)
def generate_on_batch(T: int, examples: jax.Array, *, model: BPDeconvDecoder):
    model.eval()

    return forward(examples, model=model)


@pxf.jit(static_argnums=0)
def eval_on_batch(T: int, examples: jax.Array, *, model: BPDeconvDecoder):
    pred = jnp.clip(generate_on_batch(T, examples, model=model), 0.0, 1.0)

    mse_loss = jnp.square(pred.flatten() - examples.flatten()).mean()

    return mse_loss, pred


def train(dl, T, *, model: BPDeconvDecoder, optim_w: pxu.Optim):
    for x, y in dl:
        x = x.transpose(0, 3, 1, 2)
        train_on_batch(T, x, model=model, optim_w=optim_w)


def eval(dl, T, *, model: BPDeconvDecoder):
    losses = []

    for x, y in dl:
        x = x.transpose(0, 3, 1, 2)
        e, y_hat = eval_on_batch(T, x, model=model)
        losses.append(e)

    return np.mean(losses)


def run_experiment(
    *,
    num_layers: int = 2,
    internal_state_dim: tuple[int, int, int] = (8, 8, 8),
    kernel_size: int = 5,
    batch_size: int = 500,
    epochs: int = 15,
    T: int = 15,
    optim_x_lr: float = 3e-2,
    optim_x_momentum: float = 0.0,
    optim_w_lr: float = 1e-3,
    optim_w_b1: float = 0.9,
    optim_w_b2: float = 0.999,
    num_sample_images: int = 10,
    checkpoint_dir: Path | None = None,
) -> float:
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = load_cifar10()

    input_dim = internal_state_dim
    output_dim = (3, *train_dataset[0]["img"].size)
    bottleneck_dim = 128

    model = BPDeconvDecoder(
        input_dim,
        bottleneck_dim,
        act_fn=jax.nn.tanh,
        output_act_fn=lambda x: x,
    )

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jnp.zeros((batch_size, *output_dim)), model=model)

    optim_w = pxu.Optim(
        optax.adamw(learning_rate=optim_w_lr, b1=optim_w_b1, b2=optim_w_b2), pxu.Mask(pxnn.LayerParam)(model)
    )

    if len(train_dataset) % batch_size != 0 or len(test_dataset) % batch_size != 0:
        raise ValueError("The dataset size must be divisible by the batch size.")

    test_losses = []
    for epoch in range(epochs):
        train(get_batches(train_dataset, batch_size), T=T, model=model, optim_w=optim_w)
        mse_loss = eval(get_batches(test_dataset, batch_size), T=T, model=model)
        test_losses.append(mse_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Test Loss: {mse_loss:.4f}")

    def predictor(images):
        model.clear_params(pxc.VodeParam)
        model.clear_params(pxc.VodeParam.Cache)
        images = images.transpose(0, 3, 1, 2)
        preds = generate_on_batch(T, images, model=model)
        preds = preds.transpose(0, 2, 3, 1)
        return preds

    if checkpoint_dir is not None:
        reconstruct_image(list(range(num_sample_images)), predictor, test_dataset, checkpoint_dir / "images")

    return min(test_losses)


if __name__ == "__main__":
    run_experiment(checkpoint_dir=Path("results/bp_deconv"))
