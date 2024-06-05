from typing import Callable, Union, Sequence
import math
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

from conv_transpose_layer import ConvTranspose

sys.path.insert(0, "../")
from data_utils import get_vision_dataloaders, reconstruct_image, seed_everything  # noqa: E402

sys.path.pop(0)

# 0 - 0.0068
# 1 - 0.0069 diverged
# 2 - 0.0069
# 3 - 0.0069


def seed_pcax_and_everything(seed: int | None = None):
    if seed is None:
        seed = 0
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


class PCDeconvDecoder(pxc.EnergyModule):
    def __init__(
        self,
        *,
        num_layers: int,
        input_dim: tuple[int, int, int],
        output_dim: tuple[int, int, int],
        out_channels_per_layer: list[int] | None = None,
        kernel_size: Union[int, Sequence[int]],
        act_fn: Callable[[jax.Array], jax.Array],
        output_act_fn: Callable[[jax.Array], jax.Array] = lambda x: x,
        channel_last: bool = True,
    ):
        super().__init__()
        self.act_fn = px.static(act_fn)
        self.output_act_fn = px.static(output_act_fn)
        self.channel_last = px.static(channel_last)

        input_spatial_dim: np.array
        output_spatial_dim: np.array
        input_channels: int
        output_channels: int
        if self.channel_last.get():
            input_spatial_dim = np.array(input_dim[:-1])
            output_spatial_dim = np.array(output_dim[:-1])
            input_channels = input_dim[-1]
            output_channels = output_dim[-1]
        else:
            input_spatial_dim = np.array(input_dim[1:])
            output_spatial_dim = np.array(output_dim[1:])
            input_channels = input_dim[0]
            output_channels = output_dim[0]

        input_dim = (input_channels, *input_spatial_dim)
        output_dim = (output_channels, *output_spatial_dim)

        spatial_scale = output_spatial_dim / input_spatial_dim
        if np.any(spatial_scale % 1 != 0):
            raise ValueError(
                "scale=(output_dim/input_dim) must be an integer "
                f"input_dim: {input_dim}, output_dim: {output_dim}, scale: {spatial_scale}"
            )

        step_scale = spatial_scale ** (1 / num_layers)
        if np.any(step_scale % 1 != 0):
            raise ValueError(
                "The scale=(output_dim/input_dim) must be a power of the stride number: scale = stride^num_layers. "
                f"Scale: {spatial_scale}, num_layers: {num_layers}, stride: {step_scale}"
            )
        step_scale = step_scale.astype(np.int32)

        if out_channels_per_layer:
            if len(out_channels_per_layer) != num_layers:
                raise ValueError(
                    "out_channels_per_layer must be equal to the number of layers. "
                    f"num_layers: {num_layers}, channels_per_layer: {out_channels_per_layer}"
                )
            if out_channels_per_layer[-1] != output_channels:
                raise ValueError(
                    "The number of channels in the last layer must be equal to the number of output channels. "
                    f"output_channels: {output_channels}, channels_per_layer[-1]: {out_channels_per_layer[-1]}"
                )
        else:
            channel_diff = output_channels - input_channels
            if channel_diff >= 0:
                raise ValueError(
                    "The number of input channels must be greater than the number of output channels. "
                    f"input_channels: {input_channels}, output_channels: {output_channels}"
                )
            step_channel_diff = channel_diff // num_layers
            out_channels_per_layer = [
                (input_channels + i * step_channel_diff) if i < num_layers else output_channels
                for i in range(1, num_layers + 1)
            ]

        input_dims: list[tuple[int, int, int]] = [input_dim]
        output_dims: list[tuple[int, int, int]] = []
        for i in range(num_layers):
            inp = input_dims[i]
            output_dims.append(
                (
                    out_channels_per_layer[i],
                    inp[1] * step_scale[0],
                    inp[2] * step_scale[1],
                )
            )
            if i < num_layers - 1:
                input_dims.append(output_dims[-1])
        assert len(input_dims) == len(output_dims)
        assert output_dims[-1] == output_dim

        logging.info(f"Shape transform: {input_dim} -> {' -> '.join(str(dim) for dim in output_dims)}")

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.layers = []

        for layer_input, layer_output in zip(input_dims, output_dims):
            paddings = [
                _calculate_padding_and_output_padding(
                    input_dim=layer_input[i + 1],
                    output_dim=layer_output[i + 1],
                    stride=step_scale[i],
                    kernel_size=kernel_size[i],
                )
                for i in range(2)
            ]

            padding, output_padding = zip(*paddings)

            expected_output = tuple(
                step_scale[i] * (layer_input[i + 1] - 1) + kernel_size[i] - 2 * padding[i] + output_padding[i]
                for i in range(2)
            )
            assert expected_output == tuple(layer_output[1:])

            self.layers.append(
                ConvTranspose(
                    num_spatial_dims=2,
                    in_channels=layer_input[0],
                    out_channels=layer_output[0],
                    kernel_size=kernel_size,
                    stride=(step_scale[0], step_scale[1]),
                    padding=padding,
                    output_padding=output_padding,
                )
            )

        self.vodes = [
            # pxc.Vode(
            #     (bottleneck_dim,),
            #     energy_fn=pxc.zero_energy,
            #     ruleset={pxc.STATUS.INIT: ("h, u <- u:to_zero",)},
            #     tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros(n.shape.get())},
            # ),
            # pxc.Vode(
            #     input_dim,
            #     ruleset={pxc.STATUS.INIT: ("h, u <- u:to_zero",), STATUS_FORWARD: ("h -> u",)},
            #     tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros_like(v)},
            # ),
            pxc.Vode(
                input_dim,
                energy_fn=pxc.zero_energy,
                ruleset={pxc.STATUS.INIT: ("h, u <- u:to_zero",)},
                tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros(n.shape)},
            )
        ]
        for layer_output in output_dims:
            self.vodes.append(
                pxc.Vode(
                    layer_output,
                    ruleset={pxc.STATUS.INIT: ("h, u <- u:to_zero",), STATUS_FORWARD: ("h -> u",)},
                    tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros_like(v)},
                )
            )
        self.vodes[-1].h.frozen = True

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
            if self.channel_last.get():
                example = example.transpose(2, 0, 1)
            self.vodes[-1].set("h", example)

        pred = self.vodes[-1].get("u")
        if self.channel_last.get():
            pred = pred.transpose(1, 2, 0)

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

        if self.channel_last.get():
            x = x.transpose(1, 2, 0)
        return x

    @property
    def internal_state(self) -> jax.Array:
        return self.vodes[0].get("h")


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
def forward(example: jax.Array | None = None, *, model: PCDeconvDecoder) -> jax.Array:
    return model(example=example)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=0, out_axes=0)
def generate(internal_state: jax.Array, *, model: PCDeconvDecoder) -> jax.Array:
    return model.generate(internal_state=internal_state)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=0, out_axes=(None, 0), axis_name="batch")
def energy(example: jax.Array, *, model: PCDeconvDecoder) -> jax.Array:
    y_ = model(example=example)
    return jax.lax.pmean(model.energy().sum(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch_pc(T: int, examples: jax.Array, *, model: PCDeconvDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
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
def train_on_batch_ipc(T: int, examples: jax.Array, *, model: PCDeconvDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
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
def generate_on_batch(T: int, examples: jax.Array, *, model: PCDeconvDecoder, optim_h: pxu.Optim):
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
def eval_on_batch(T: int, examples: jax.Array, *, model: PCDeconvDecoder, optim_h: pxu.Optim):
    pred = generate_on_batch(T, examples, model=model, optim_h=optim_h)

    mse_loss = jnp.square(pred.flatten() - examples.flatten()).mean()

    return mse_loss, pred


def train(
    dl, T, *, model: PCDeconvDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim, batch_size: int, use_ipc: bool = False
):
    for x, y in dl:
        if x.shape[0] != batch_size:
            logging.warning(f"Skipping batch of size {x.shape[0]} that's not equal to the batch size {batch_size}.")
            continue
        if use_ipc:
            train_on_batch_ipc(T, x, model=model, optim_w=optim_w, optim_h=optim_h)
        else:
            train_on_batch_pc(T, x, model=model, optim_w=optim_w, optim_h=optim_h)


def eval(dl, T, *, model: PCDeconvDecoder, optim_h: pxu.Optim, batch_size: int):
    losses = []

    for x, y in dl:
        if x.shape[0] != batch_size:
            logging.warning(f"Skipping batch of size {x.shape[0]} that's not equal to the batch size {batch_size}.")
            continue
        e, y_hat = eval_on_batch(T, x, model=model, optim_h=optim_h)
        losses.append(e)

    return np.mean(losses)


def run_experiment(
    *,
    dataset_name: str = "cifar10",
    num_layers: int = 3,
    internal_state_dim: tuple[int, int, int] = (8, 4, 4),
    kernel_size: int = 7,
    act_fn: str | None = "tanh",
    output_act_fn: str | None = None,
    batch_size: int = 200,
    epochs: int = 30,
    T: int = 20,
    use_ipc: bool = False,
    optim_x_lr: float = 0.012339577360613845,
    optim_x_momentum: float = 0.1,
    optim_w_name: str = "adamw",
    optim_w_lr: float = 0.0007642161267055484,
    optim_w_wd: float = 3.511389190873198e-05,
    optim_w_b1: float = 0.9,
    optim_w_b2: float = 0.999,
    optim_w_momentum: float = 0.1,
    num_sample_images: int = 10,
    checkpoint_dir: Path | None = None,
    seed: int | None = None,
) -> float:
    seed_pcax_and_everything(seed)

    # Channel first: (batch, channel, height, width)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_vision_dataloaders(dataset_name=dataset_name, batch_size=batch_size, should_normalize=False)

    input_dim = internal_state_dim
    output_dim = dataset.train_dataset[0][0].shape

    model = PCDeconvDecoder(
        num_layers=num_layers,
        input_dim=input_dim,
        output_dim=output_dim,
        out_channels_per_layer=[8, 5, 3],
        kernel_size=kernel_size,
        act_fn=ACTIVATION_FUNCS[act_fn],
        output_act_fn=ACTIVATION_FUNCS[output_act_fn],
        channel_last=False,
    )

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jnp.zeros((batch_size, *output_dim)), model=model)

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
        model.clear_params(pxc.VodeParam)
        model.clear_params(pxc.VodeParam.Cache)
        return generate_on_batch(T, images, model=model, optim_h=optim_h)

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
    run_experiment(checkpoint_dir=Path("results/pc_deconv"))
