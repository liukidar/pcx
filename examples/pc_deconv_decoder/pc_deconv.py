from typing import Callable, Union, Sequence
import math

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
from data_utils import load_cifar10, get_batches


STATUS_FORWARD = "forward"


class PCDeconvDecoder(pxc.EnergyModule):
    def __init__(
        self,
        input_dim: tuple[int, int, int],
        output_dim: tuple[int, int, int],
        num_layers: int,
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

        channel_diff = output_channels - input_channels
        if channel_diff >= 0:
            raise ValueError(
                "The number of input channels must be greater than the number of output channels. "
                f"input_channels: {input_channels}, output_channels: {output_channels}"
            )

        step_scale = spatial_scale ** (1 / num_layers)
        if np.any(step_scale % 1 != 0):
            raise ValueError(
                "The scale=(output_dim/input_dim) must be a power of the stride number: scale = stride^num_layers. "
                f"Scale: {spatial_scale}, num_layers: {num_layers}, stride: {step_scale}"
            )
        step_scale = step_scale.astype(np.int32)

        step_channel_diff = channel_diff // num_layers

        input_dims: list[tuple[int, int, int]] = [input_dim]
        output_dims: list[tuple[int, int, int]] = []
        for i in range(num_layers):
            inp = input_dims[i]
            output_dims.append(
                (
                    inp[0] + step_channel_diff if i < num_layers - 1 else output_channels,
                    inp[1] * step_scale[0],
                    inp[2] * step_scale[1],
                )
            )
            if i < num_layers - 1:
                input_dims.append(output_dims[-1])
        assert len(input_dims) == len(output_dims)
        assert output_dims[-1] == output_dim

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

    def __call__(self, example: jax.Array | None = None, internal_state: jax.Array | None = None):
        # The defined ruleset for the first node is to set the hidden state to zero,
        # independent of the input, so we always pass '-1'.
        x = self.vodes[0](-1)
        if internal_state is not None:
            x = internal_state

        for i, layer in enumerate(self.layers):
            act_fn = self.act_fn if i < len(self.layers) - 1 else self.output_act_fn
            x = act_fn(layer(x))
            x = self.vodes[i + 1](x)

        if example is not None:
            if self.channel_last.get():
                example = example.transpose(2, 0, 1)
            self.vodes[-1].set("h", example.flatten())

        pred = self.vodes[-1].get("u")
        if self.channel_last.get():
            pred = pred.transpose(1, 2, 0)
        return pred


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
def forward(example: jax.Array, *, model: PCDeconvDecoder) -> jax.Array:
    return model(example=example)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=0, out_axes=(None, 0), axis_name="batch")
def energy(example: jax.Array, *, model: PCDeconvDecoder) -> jax.Array:
    y_ = model(example=example)
    return jax.lax.pmean(model.energy().sum(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch(T: int, examples: jax.Array, *, model: PCDeconvDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
    model.train()

    inference_step = pxf.value_and_grad(
        pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]), has_aux=True
    )(energy)

    learning_step = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(energy)

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(examples, model=model)

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = inference_step(model=model)

        optim_h.step(model, g["model"])

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = learning_step(model=model)
    optim_w.step(model, g["model"])


@pxf.jit(static_argnums=0)
def eval_on_batch(T: int, examples: jax.Array, *, model: PCDeconvDecoder, optim_h: pxu.Optim):
    model.eval()

    inference_step = pxf.value_and_grad(
        pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]), has_aux=True
    )(energy)

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(examples, model=model)

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = inference_step(model=model)

        optim_h.step(model, g["model"])

    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        x_hat = forward(None, model=model)

    mse_loss = jnp.square(jnp.clip(x_hat.flatten(), 0.0, 1.0) - examples.flatten()).mean()

    return mse_loss, x_hat


def train(dl, T, *, model: PCDeconvDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
    for x, y in dl:
        train_on_batch(T, x, model=model, optim_w=optim_w, optim_h=optim_h)


def eval(dl, T, *, model: PCDeconvDecoder, optim_h: pxu.Optim):
    losses = []

    for x, y in dl:
        e, y_hat = eval_on_batch(T, x, model=model, optim_h=optim_h)
        losses.append(e)

    return np.mean(e)


def main():
    input_dim = (64, 4, 4)
    output_dim = (3, 32, 32)
    model = PCDeconvDecoder(
        input_dim,
        output_dim,
        num_layers=3,
        kernel_size=3,
        act_fn=jax.nn.relu,
        output_act_fn=jax.nn.sigmoid,
        channel_last=False,
    )

    batch_size = 128
    nm_epochs = 3

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jnp.zeros((batch_size, *output_dim)), model=model)

    optim_h = pxu.Optim(optax.sgd(3e-2 * batch_size), pxu.Mask(pxc.VodeParam)(model))
    optim_w = pxu.Optim(optax.adamw(5e-4), pxu.Mask(pxnn.LayerParam)(model))

    train_dataset, test_dataset = load_cifar10()

    for e in range(nm_epochs):
        train(get_batches(train_dataset, batch_size), T=20, model=model, optim_w=optim_w, optim_h=optim_h)
        mse_loss = eval(get_batches(test_dataset, batch_size), T=20, model=model, optim_h=optim_h)
        print(f"Epoch {e + 1}/{nm_epochs} - Test Loss: {mse_loss:.4f}")


if __name__ == "__main__":
    main()


# import equinox as eqx

# c = eqx.nn.ConvTranspose(
#     num_spatial_dims=2,
#     in_channels=64,
#     out_channels=3,
#     kernel_size=3,
#     stride=2,
#     padding=1,
#     output_padding=1,
#     key=jax.random.PRNGKey(0),
# )
# input = jnp.ones((64, 4, 4))
# output = c(input)
# print(output.shape)


# # Example usage
# input_dim = 4
# desired_output_dim = 8
# stride = 2
# kernel_size = 3

# padding, output_padding = calculate_padding_and_output_padding(input_dim, desired_output_dim, stride, kernel_size)
# print(f"Required padding: {padding}")
# print(f"Required output_padding: {output_padding}")
