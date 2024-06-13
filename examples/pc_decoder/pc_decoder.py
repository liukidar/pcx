from typing import Callable
from pathlib import Path
import logging
import sys

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

sys.path.insert(0, "../")
from data_utils import get_vision_dataloaders, reconstruct_image, seed_everything  # noqa: E402

sys.path.pop(0)


def seed_pcax_and_everything(seed: int | None = None):
    if seed is None:
        seed = 4
    RKG.seed(seed)
    seed_everything(seed)


logging.basicConfig(level=logging.INFO)


STATUS_FORWARD = "forward"
ACTIVATION_FUNCS = {
    None: lambda x: x,
    "relu": jax.nn.relu,
    "leaky_relu": jax.nn.leaky_relu,
    "gelu": jax.nn.gelu,
    "tanh": jax.nn.tanh,
    "hard_tanh": jax.nn.hard_tanh,
    "sigmoid": jax.nn.sigmoid,
}


class PCDecoder(pxc.EnergyModule):
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

        self.layers = []
        for layer_input, layer_output in zip(layer_dims[:-1], layer_dims[1:]):
            self.layers.append(pxnn.Linear(layer_input, layer_output))

        self.vodes = [
            pxc.Vode(
                (layer_dims[0],),
                energy_fn=pxc.zero_energy,
                ruleset={pxc.STATUS.INIT: ("h, u <- u:to_zero",)},
                tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros(n.shape.get())},
            )
        ]
        for layer_output in layer_dims[1:]:
            self.vodes.append(
                pxc.Vode(
                    (layer_output,),
                    ruleset={pxc.STATUS.INIT: ("h, u <- u:to_zero",), STATUS_FORWARD: ("h -> u",)},
                    tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros_like(v)},
                )
            )

        self.vodes[-1].h.frozen = True

        logging.info(f"Shape transform: {layer_dims}")

    def __call__(self, example: jax.Array | None = None, internal_state: jax.Array | None = None) -> jax.Array:
        # The defined ruleset for the first node is to set the hidden state to zero,
        # independent of the input, so we always pass '-1'.
        x = self.vodes[0](-1)
        if internal_state is not None:
            x = internal_state

        for i, layer in enumerate(self.layers):
            act_fn = self.act_fn if i < len(self.layers) - 1 else self.output_act_fn
            x = act_fn(layer(x))
            # jax.debug.print(
            #     "Layer {i}: {shape}: {nan} [{min}, {mean}, {max}]",
            #     i=i,
            #     shape=x.shape,
            #     nan=jnp.any(jnp.isnan(x)),
            #     mean=x.mean(),
            #     min=x.min(),
            #     max=x.max(),
            # )
            x = self.vodes[i + 1](x)
            # jax.debug.print(
            #     "Vode {i}: {shape}: {nan} [{min}, {mean}, {max}]",
            #     i=i,
            #     shape=x.shape,
            #     nan=jnp.any(jnp.isnan(x)),
            #     mean=x.mean(),
            #     min=x.min(),
            #     max=x.max(),
            # )

        if example is not None:
            self.vodes[-1].set("h", example)

        pred = self.vodes[-1].get("u")

        assert example is None or pred.shape == example.shape
        # jax.debug.print(
        #     "__call__:\n"
        #     "P: m={pred_mean} s={pred_std} min={pred_min} max={pred_max}\n"
        #     "E: m={example_mean} s={example_std} min={example_min} max={example_max}\n",
        #     pred_mean=pred.mean(axis=(1, 2)),
        #     pred_std=pred.std(axis=(1, 2)),
        #     pred_min=pred.min(axis=(1, 2)),
        #     pred_max=pred.max(axis=(1, 2)),
        #     example_mean=example.mean(axis=(1, 2)),
        #     example_std=example.std(axis=(1, 2)),
        #     example_min=example.min(axis=(1, 2)),
        #     example_max=example.max(axis=(1, 2)),
        # )
        return pred

    def generate(self, internal_state: jax.Array | None = None) -> jax.Array:
        x = internal_state
        if x is None:
            x = self.internal_state

        for i, layer in enumerate(self.layers):
            act_fn = self.act_fn if i < len(self.layers) - 1 else self.output_act_fn
            x = act_fn(layer(x))

        return x

    @property
    def internal_state(self) -> jax.Array:
        return self.vodes[0].get("h")


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=0, out_axes=0)
def forward(example: jax.Array | None = None, *, model: PCDecoder) -> jax.Array:
    return model(example=example)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=0, out_axes=0)
def generate(internal_state: jax.Array, *, model: PCDecoder) -> jax.Array:
    return model.generate(internal_state=internal_state)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=0, out_axes=(None, 0), axis_name="batch")
def energy(example: jax.Array, *, model: PCDecoder) -> jax.Array:
    y_ = model(example=example)
    return jax.lax.pmean(model.energy().sum(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch_pc(T: int, examples: jax.Array, *, model: PCDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
    model.train()

    inference_step = pxf.value_and_grad(
        pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]), has_aux=True
    )(energy)

    learning_step = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(energy)

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(examples, model=model)

    optim_h.init(pxu.Mask(pxc.VodeParam)(model))

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = inference_step(examples, model=model)

        optim_h.step(model, g["model"], scale_by_batch_size=True)

    optim_h.clear()

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = learning_step(examples, model=model)
    optim_w.step(model, g["model"])


@pxf.jit(static_argnums=0)
def train_on_batch_ipc(T: int, examples: jax.Array, *, model: PCDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
    model.train()

    step = pxf.value_and_grad(
        pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True) | pxu.m(pxnn.LayerParam), [False, True]), has_aux=True
    )(energy)

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(examples, model=model)

    optim_h.init(pxu.Mask(pxc.VodeParam)(model))

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = step(examples, model=model)

        optim_h.step(model, g["model"], scale_by_batch_size=True)
        optim_w.step(model, g["model"])
    optim_h.clear()


@pxf.jit(static_argnums=0)
def generate_on_batch(T: int, examples: jax.Array, *, model: PCDecoder, optim_h: pxu.Optim):
    model.eval()

    inference_step = pxf.value_and_grad(
        pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]), has_aux=True
    )(energy)

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(examples, model=model)

    optim_h.init(pxu.Mask(pxc.VodeParam)(model))

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = inference_step(examples, model=model)

        optim_h.step(model, g["model"], scale_by_batch_size=True)
    optim_h.clear()
    pred = generate(model.internal_state, model=model)

    assert pred.shape == examples.shape
    # jax.debug.print(
    #     "generate_on_batch:\n"
    #     "P: m={pred_mean} s={pred_std} min={pred_min} max={pred_max}\n"
    #     "E: m={examples_mean} s={examples_std} min={examples_min} max={examples_max}\n",
    #     # pred_shape=pred.shape,
    #     pred_mean=pred.mean(axis=(0, 2, 3)),
    #     pred_std=pred.std(axis=(0, 2, 3)),
    #     pred_min=pred.min(axis=(0, 2, 3)),
    #     pred_max=pred.max(axis=(0, 2, 3)),
    #     # examples_shape=examples.shape,
    #     examples_mean=examples.mean(axis=(0, 2, 3)),
    #     examples_std=examples.std(axis=(0, 2, 3)),
    #     examples_min=examples.min(axis=(0, 2, 3)),
    #     examples_max=examples.max(axis=(0, 2, 3)),
    # )

    return pred


@pxf.jit(static_argnums=0)
def eval_on_batch(T: int, examples: jax.Array, *, model: PCDecoder, optim_h: pxu.Optim):
    pred = generate_on_batch(T, examples, model=model, optim_h=optim_h)

    mse_loss = jnp.square(pred.flatten() - examples.flatten()).mean()

    return mse_loss, pred


def train(dl, T, *, model: PCDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim, batch_size: int, use_ipc: bool = False):
    for x, y in dl:
        if x.shape[0] != batch_size:
            logging.warning(f"Skipping batch of size {x.shape[0]} that's not equal to the batch size {batch_size}.")
            continue
        x = x.reshape(x.shape[0], -1)
        if use_ipc:
            train_on_batch_ipc(T, x, model=model, optim_w=optim_w, optim_h=optim_h)
        else:
            train_on_batch_pc(T, x, model=model, optim_w=optim_w, optim_h=optim_h)


def eval(dl, T, *, model: PCDecoder, optim_h: pxu.Optim, batch_size: int):
    losses = []

    for x, y in dl:
        if x.shape[0] != batch_size:
            logging.warning(f"Skipping batch of size {x.shape[0]} that's not equal to the batch size {batch_size}.")
            continue
        x = x.reshape(x.shape[0], -1)
        e, y_hat = eval_on_batch(T, x, model=model, optim_h=optim_h)
        losses.append(e)

    return np.mean(losses)


def run_experiment(
    *,
    dataset_name: str = "fashion_mnist",
    layer_dims: list[int] = [64, 128, 128, 784],
    act_fn: str | None = "hard_tanh",
    output_act_fn: str | None = None,
    batch_size: int = 200,
    epochs: int = 30,
    T: int = 20,
    use_ipc: bool = False,
    optim_x_lr: float = 0.011910178174048628,
    optim_x_momentum: float = 0.45,
    optim_w_name: str = "adamw",
    optim_w_lr: float = 0.0005149979608369072,
    optim_w_wd: float = 7.382135878282121e-05,
    optim_w_b1: float = 0.9,
    optim_w_b2: float = 0.999,
    optim_w_momentum: float = 0.0,
    num_sample_images: int = 10,
    checkpoint_dir: Path | None = None,
    seed: int | None = None,
) -> float:
    seed_pcax_and_everything(seed)

    # Channel first: (batch, channel, height, width)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_vision_dataloaders(dataset_name=dataset_name, batch_size=batch_size, should_normalize=False)

    output_dim = layer_dims[-1]

    model = PCDecoder(
        layer_dims=layer_dims,
        act_fn=ACTIVATION_FUNCS[act_fn],
        output_act_fn=ACTIVATION_FUNCS[output_act_fn],
    )

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jnp.zeros((batch_size, output_dim)), model=model)

    optim_h = pxu.Optim(optax.sgd(learning_rate=optim_x_lr, momentum=optim_x_momentum))
    if optim_w_name == "adamw":
        optim_w = pxu.Optim(
            optax.adamw(learning_rate=optim_w_lr, weight_decay=optim_w_wd, b1=optim_w_b1, b2=optim_w_b2),
            pxu.Mask(pxnn.LayerParam)(model),
        )
    elif optim_w_name == "sgd":
        optim_w = pxu.Optim(
            optax.sgd(learning_rate=optim_w_lr, momentum=optim_w_momentum), pxu.Mask(pxnn.LayerParam)(model)
        )
    else:
        raise ValueError(f"Unknown optimizer name: {optim_w_name}")

    # if len(dataset.train_dataset) % batch_size != 0 or len(dataset.test_dataset) % batch_size != 0:
    #     raise ValueError("The dataset size must be divisible by the batch size.")

    model_save_dir: Path | None = checkpoint_dir / dataset_name / "best_model" if checkpoint_dir is not None else None
    model_saved: bool = False
    if model_save_dir is not None:
        model_save_dir.mkdir(parents=True, exist_ok=True)

    print("Training...")

    best_loss: float | None = None
    test_losses: list[float] = []
    for epoch in range(epochs):
        train(
            dataset.train_dataloader,
            T=T,
            model=model,
            optim_w=optim_w,
            optim_h=optim_h,
            batch_size=batch_size,
            use_ipc=use_ipc,
        )
        mse_loss = eval(dataset.test_dataloader, T=T, model=model, optim_h=optim_h, batch_size=batch_size)
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
        pred = generate_on_batch(T, images, model=model, optim_h=optim_h)
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
    run_experiment(checkpoint_dir=Path("results/pc_decoder"))
