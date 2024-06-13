from typing import Callable
from pathlib import Path
import sys
import logging
import argparse

# Core dependencies
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf

# pcax
import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.utils as pxu
import pcax.functional as pxf
from pcax import RKG

sys.path.insert(0, "../../")
from data_utils import get_vision_dataloaders, reconstruct_image, seed_everything, get_config_value  # noqa: E402

sys.path.pop(0)


def seed_pcax_and_everything(seed: int | None = None):
    if seed is None:
        seed = 0
    RKG.seed(seed)
    seed_everything(seed)


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


class BPDecoder(pxc.EnergyModule):
    def __init__(
        self,
        *,
        layer_dims: list[int],
        act_fn: Callable[[jax.Array], jax.Array],
        output_act_fn: Callable[[jax.Array], jax.Array] = lambda x: x,
    ):
        super().__init__()
        self.act_fn = px.static(act_fn)
        self.output_act_fn = px.static(output_act_fn)

        layer_dims = layer_dims[::-1] + layer_dims[1:]

        self.layers = []
        for layer_input, layer_output in zip(layer_dims[:-1], layer_dims[1:]):
            self.layers.append(pxnn.Linear(layer_input, layer_output))

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=0, out_axes=0)
def forward(x: jax.Array | None = None, *, model: BPDecoder) -> jax.Array:
    return model(x)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=0, out_axes=(None, 0), axis_name="batch")
def loss(example: jax.Array, *, model: BPDecoder) -> jax.Array:
    pred = model(example)
    assert pred.shape == example.shape
    mse_loss = jnp.square(pred.flatten() - example.flatten()).mean()
    return jax.lax.pmean(mse_loss, "batch"), pred


@pxf.jit()
def train_on_batch(examples: jax.Array, *, model: BPDecoder, optim_w: pxu.Optim):
    model.train()

    learning_step = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(loss)

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = learning_step(examples, model=model)
    optim_w.step(model, g["model"])


@pxf.jit()
def generate_on_batch(examples: jax.Array, *, model: BPDecoder):
    model.eval()

    return forward(examples, model=model)


@pxf.jit()
def eval_on_batch(examples: jax.Array, *, model: BPDecoder):
    pred = generate_on_batch(examples, model=model)

    assert pred.shape == examples.shape
    mse_loss = jnp.square(pred.flatten() - examples.flatten()).mean()

    return mse_loss, pred


def train(dl, *, model: BPDecoder, optim_w: pxu.Optim, batch_size: int):
    for x, y in dl:
        if x.shape[0] != batch_size:
            logging.warning(f"Skipping batch of size {x.shape[0]} that's not equal to the batch size {batch_size}.")
            continue
        x = x.reshape(x.shape[0], -1)
        train_on_batch(x, model=model, optim_w=optim_w)


def eval(dl, *, model: BPDecoder, batch_size: int):
    losses = []

    for x, y in dl:
        if x.shape[0] != batch_size:
            logging.warning(f"Skipping batch of size {x.shape[0]} that's not equal to the batch size {batch_size}.")
            continue
        x = x.reshape(x.shape[0], -1)
        e, y_hat = eval_on_batch(x, model=model)
        losses.append(e)

    return np.mean(losses)


def run_experiment(
    *,
    dataset_name: str,
    layer_dims: list[int],
    act_fn: str | None,
    output_act_fn: str | None,
    batch_size: int,
    epochs: int,
    optim_w_name: str,
    optim_w_lr: float,
    optim_w_wd: float,
    optim_w_momentum: float,
    num_sample_images: int = 10,
    checkpoint_dir: Path | None = None,
    seed: int | None = None,
) -> float:
    seed_pcax_and_everything(seed)

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_vision_dataloaders(dataset_name=dataset_name, batch_size=batch_size, should_normalize=False)

    output_dim = layer_dims[-1]

    model = BPDecoder(
        layer_dims=layer_dims,
        act_fn=ACTIVATION_FUNCS[act_fn],
        output_act_fn=ACTIVATION_FUNCS[output_act_fn],
    )

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jnp.zeros((batch_size, output_dim)), model=model)

    if optim_w_name == "adamw":
        optim_w = pxu.Optim(
            optax.adamw(learning_rate=optim_w_lr, weight_decay=optim_w_wd),
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
        if np.isnan(mse_loss):
            logging.warning("Model diverged. Stopping training.")
            break
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
        batch_size, channel, height, width = images.shape
        images = images.reshape(batch_size, -1)
        model.clear_params(pxc.VodeParam)
        model.clear_params(pxc.VodeParam.Cache)
        pred = generate_on_batch(images, model=model)
        pred = pred.reshape(batch_size, channel, height, width)
        return pred

    if checkpoint_dir is not None:
        reconstruct_image(
            list(range(num_sample_images)),
            predictor,
            dataset.test_dataset,
            dataset.image_restore,
            checkpoint_dir / dataset_name / "images",
        )

    return min(test_losses) if test_losses else np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/bp_fashionmnist_adamw_hypertune.yaml", help="Path to the config file."
    )

    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    run_experiment(
        dataset_name=get_config_value(config, "dataset_name"),
        seed=get_config_value(config, "seed", required=False),
        layer_dims=get_config_value(config, "hp/layer_dims"),
        act_fn=get_config_value(config, "hp/act_fn"),
        output_act_fn=get_config_value(config, "hp/output_act_fn"),
        batch_size=get_config_value(config, "hp/batch_size"),
        epochs=get_config_value(config, "hp/epochs"),
        optim_w_name=get_config_value(config, "hp/optim/w/name"),
        optim_w_lr=get_config_value(config, "hp/optim/w/lr"),
        optim_w_wd=get_config_value(config, "hp/optim/w/wd"),
        optim_w_momentum=get_config_value(config, "hp/optim/w/momentum"),
        checkpoint_dir=Path("results/bp_decoder"),
    )
