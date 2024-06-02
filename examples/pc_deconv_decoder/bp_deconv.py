from typing import Callable
from pathlib import Path
import math
import sys
import logging

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
from pcax import RKG

from conv_transpose_layer import ConvTranspose

sys.path.insert(0, "../")
from data_utils import get_vision_dataloaders, reconstruct_image, seed_everything  # noqa: E402

sys.path.pop(0)

# 0 - 0.0067
# 1 - 0.0058
# 2 - 0.0067
# 3 - 0.0057
RKG.seed(3)
seed_everything(3)


logging.basicConfig(level=logging.INFO)


ACTIVATION_FUNCS = {
    None: lambda x: x,
    "relu": jax.nn.relu,
    "leaky_relu": jax.nn.leaky_relu,
    "gelu": jax.nn.gelu,
    "tanh": jax.nn.tanh,
    "hard_tanh": jax.nn.hard_tanh,
    "sigmoid": jax.nn.sigmoid,
}


class BPDeconvDecoder(pxc.EnergyModule):
    def __init__(
        self,
        *,
        kernel_size: int = 5,
        act_fn: Callable[[jax.Array], jax.Array],
        output_act_fn: Callable[[jax.Array], jax.Array] = lambda x: x,
    ):
        super().__init__()
        self.act_fn = px.static(act_fn)
        self.output_act_fn = px.static(output_act_fn)

        conv_padding = (kernel_size - 1) // 2
        input_dims = [4, 8, 16]
        output_dims = [8, 16, 32]
        conv_transpose_paddings = []
        for input_dim, output_dim in zip(input_dims, output_dims):
            padding, output_padding = _calculate_padding_and_output_padding(
                input_dim=input_dim, output_dim=output_dim, stride=2, kernel_size=kernel_size
            )
            conv_transpose_paddings.append({"padding": padding, "output_padding": output_padding})

        self.layers = [
            pxnn.Conv2d(
                3, 5, kernel_size=(kernel_size, kernel_size), stride=(2, 2), padding=conv_padding
            ),  # (5, 16, 16)
            self.act_fn,
            pxnn.Conv2d(5, 8, kernel_size=(kernel_size, kernel_size), stride=(2, 2), padding=conv_padding),  # (8, 8, 8)
            self.act_fn,
            pxnn.Conv2d(8, 8, kernel_size=(kernel_size, kernel_size), stride=(2, 2), padding=conv_padding),  # (8, 4, 4)
            self.act_fn,
            ConvTranspose(
                2, 8, 8, kernel_size=(kernel_size, kernel_size), stride=(2, 2), **conv_transpose_paddings[0]
            ),  # (8, 8, 8)
            self.act_fn,
            ConvTranspose(
                2, 8, 5, kernel_size=(kernel_size, kernel_size), stride=(2, 2), **conv_transpose_paddings[1]
            ),  # (5, 16, 16)
            self.act_fn,
            ConvTranspose(
                2, 5, 3, kernel_size=(kernel_size, kernel_size), stride=(2, 2), **conv_transpose_paddings[2]
            ),  # (3, 32, 32)
            self.output_act_fn,
        ]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


def _calculate_padding_and_output_padding(
    *, input_dim: int, output_dim: int, stride: int, kernel_size: int
) -> tuple[int, int]:
    """
    Calculate the padding and output_padding required for a ConvTranspose layer to achieve the desired output dimension.

    Parameters:
    input_dim (int): The size of the input dimension (height or width).
    output_dim (int): The desired size of the output dimension (height or width).
    stride (int): The stride of the convolution.
    kernel_size (int): The size of the convolution kernel.

    Returns:
    tuple: The required padding and output_padding to achieve the desired output dimension.
    """
    no_padding_output_dim = (input_dim - 1) * stride + kernel_size

    padding = math.ceil(max(no_padding_output_dim - output_dim, 0) / 2)
    output_padding = max(output_dim - (no_padding_output_dim - 2 * padding), 0)

    assert no_padding_output_dim - 2 * padding + output_padding == output_dim

    return padding, output_padding


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=0, out_axes=0)
def forward(x: jax.Array | None = None, *, model: BPDeconvDecoder) -> jax.Array:
    return model(x)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=0, out_axes=(None, 0), axis_name="batch")
def loss(example: jax.Array, *, model: BPDeconvDecoder) -> jax.Array:
    pred = model(example)
    assert pred.shape == example.shape
    mse_loss = jnp.square(pred.flatten() - example.flatten()).mean()
    return jax.lax.pmean(mse_loss, "batch"), pred


@pxf.jit()
def train_on_batch(examples: jax.Array, *, model: BPDeconvDecoder, optim_w: pxu.Optim):
    model.train()

    learning_step = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(loss)

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = learning_step(examples, model=model)
    optim_w.step(model, g["model"])


@pxf.jit()
def generate_on_batch(examples: jax.Array, *, model: BPDeconvDecoder):
    model.eval()

    return forward(examples, model=model)


@pxf.jit()
def eval_on_batch(examples: jax.Array, *, model: BPDeconvDecoder):
    pred = generate_on_batch(examples, model=model)

    assert pred.shape == examples.shape
    mse_loss = jnp.square(pred.flatten() - examples.flatten()).mean()

    return mse_loss, pred


def train(dl, *, model: BPDeconvDecoder, optim_w: pxu.Optim, batch_size: int):
    for x, y in dl:
        if x.shape[0] != batch_size:
            logging.warning(f"Skipping batch of size {x.shape[0]} that's not equal to the batch size {batch_size}.")
            continue
        train_on_batch(x, model=model, optim_w=optim_w)


def eval(dl, *, model: BPDeconvDecoder, batch_size: int):
    losses = []

    for x, y in dl:
        if x.shape[0] != batch_size:
            logging.warning(f"Skipping batch of size {x.shape[0]} that's not equal to the batch size {batch_size}.")
            continue
        e, y_hat = eval_on_batch(x, model=model)
        losses.append(e)

    return np.mean(losses)


def run_experiment(
    *,
    dataset_name: str = "cifar10",
    kernel_size: int = 7,
    act_fn: str | None = "hard_tanh",
    output_act_fn: str | None = None,
    batch_size: int = 200,
    epochs: int = 30,
    optim_w_name: str = "adamw",
    optim_w_lr: float = 0.0007958728757424726,
    optim_w_wd: float = 0.0008931102704862562,
    optim_w_b1: float = 0.9,
    optim_w_b2: float = 0.999,
    optim_w_momentum: float = 0.1,
    num_sample_images: int = 10,
    checkpoint_dir: Path | None = None,
) -> float:
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_vision_dataloaders(dataset_name=dataset_name, batch_size=batch_size, should_normalize=False)

    output_dim = dataset.train_dataset[0][0].shape

    model = BPDeconvDecoder(
        kernel_size=kernel_size,
        act_fn=ACTIVATION_FUNCS[act_fn],
        output_act_fn=ACTIVATION_FUNCS[output_act_fn],
    )

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jnp.zeros((batch_size, *output_dim)), model=model)

    if optim_w_name == "adamw":
        optim_w = pxu.Optim(
            optax.adamw(learning_rate=optim_w_lr, weight_decay=optim_w_wd, b1=optim_w_b1, b2=optim_w_b2),
            pxu.Mask(pxnn.LayerParam)(model),
        )
    elif optim_w_name == "sgd":
        optim_w = pxu.Optim(
            optax.sgd(learning_rate=optim_w_lr, momentum=optim_w_momentum),
            pxu.Mask(pxnn.LayerParam)(model),
        )
    else:
        raise ValueError(f"Unknown optimizer name: {optim_w_name}")

    # if len(dataset.train_dataset) % batch_size != 0 or len(dataset.test_dataset) % batch_size != 0:
    #     raise ValueError("The dataset size must be divisible by the batch size.")

    model_save_dir: Path | None = checkpoint_dir / dataset_name / "best_model" if checkpoint_dir is not None else None
    model_saved: bool = False
    if model_save_dir is not None:
        model_save_dir.mkdir(parents=True, exist_ok=True)

    best_loss: float | None = None
    test_losses: list[float] = []
    for epoch in range(epochs):
        train(dataset.train_dataloader, model=model, optim_w=optim_w, batch_size=batch_size)
        mse_loss = eval(dataset.test_dataloader, model=model, batch_size=batch_size)
        test_losses.append(mse_loss)
        if epochs > 1 and model_save_dir is not None and (best_loss is None or mse_loss <= best_loss):
            best_loss = mse_loss
            pxu.save_params(model, str(model_save_dir / "model"))
            model_saved = True
        print(f"Epoch {epoch + 1}/{epochs} - Test Loss: {mse_loss:.4f}")

    if model_saved:
        pxu.load_params(model, str(model_save_dir / "model"))
        logging.info(f"Loaded best model with test loss: {best_loss:.4f}")

    def predictor(images):
        model.clear_params(pxc.VodeParam)
        model.clear_params(pxc.VodeParam.Cache)
        preds = generate_on_batch(images, model=model)
        return preds

    if checkpoint_dir is not None:
        reconstruct_image(
            list(range(num_sample_images)),
            predictor,
            dataset.test_dataset,
            dataset.image_restore,
            checkpoint_dir / dataset_name / "images",
        )

    return min(test_losses)


if __name__ == "__main__":
    run_experiment(checkpoint_dir=Path("results/bp_deconv"))
