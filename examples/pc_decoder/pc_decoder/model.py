import json
import logging
import os
from typing import Callable
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pc_decoder.params import ModelParams

import pcax as px  # type: ignore
import pcax.core as pxc  # type: ignore
import pcax.nn as nn  # type: ignore
import pcax.utils as pxu  # type: ignore

activation_functions: dict[str, Callable[[jax.Array], jax.Array]] = {
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
    "sigmoid": jax.nn.sigmoid,
}


class PCDecoder(px.EnergyModule):
    def __init__(
        self,
        *,
        params: ModelParams,
        internal_state_init_fn: Callable[[pxc.RandomKeyGenerator, int], jax.Array],
    ) -> None:
        super().__init__()

        self.p = params.copy()
        self.act_fn = activation_functions[params.activation]

        def internal_node_init_fn(node: px.Node, rkg: pxc.RandomKeyGenerator) -> None:
            value = internal_state_init_fn(rkg, self.p.internal_dim)
            if not self.p.use_prior_layer:
                node.set_activation("u", value)
            node.x.value = value

        self.prior_layer: nn.Linear | None = None
        if self.p.use_prior_layer:
            # Weights are sampled from the uniform distribution
            self.prior_layer = nn.Linear(1, self.p.internal_dim)
        self.fc_layers: list[nn.Linear] = (
            [nn.Linear(self.p.internal_dim, self.p.hidden_dim)]
            + [
                nn.Linear(self.p.hidden_dim, self.p.hidden_dim)
                for _ in range(self.p.num_hidden_layers)
            ]
            + [nn.Linear(self.p.hidden_dim, self.p.output_dim)]
        )
        self.pc_nodes: list[px.Node] = [px.Node(init_fn=internal_node_init_fn)] + [
            px.Node() for _ in range(self.num_layers)
        ]

        self.pc_nodes[-1].x.frozen = True

    def __call__(
        self, example: jax.Array | None = None, internal_state: jax.Array | None = None
    ) -> jax.Array:
        if internal_state is None:
            if self.prior_layer is not None:
                internal_state = self.prior_layer(jnp.ones((1,)))
            else:
                internal_state = self.internal_state  # this might be None as well.
        # Call the internal layer so it can initialize the internal state by calling init_fn.
        x = self.pc_nodes[0](internal_state)["x"]  # type: ignore

        for i in range(self.num_layers):
            # No activation function at the last layer
            act_fn = self.act_fn if i < self.num_layers - 1 else lambda x: x
            x = self.pc_nodes[i + 1](act_fn(self.fc_layers[i](x)))["x"]

        # During training, fix target to the input
        # so that the energy encodes the difference between the prediction u and the target x.
        if example is not None:
            self.pc_nodes[-1]["x"] = example

        return self.prediction

    def feed_forward_predict(self, x: jax.Array | None = None) -> jax.Array:
        if x is None:
            x = self.internal_state
        if x is None:
            raise RuntimeError("Internal state is none.")
        for i in range(self.num_layers):
            # No activation function at the last layer
            act_fn = self.act_fn if i < self.num_layers - 1 else lambda x: x
            x = act_fn(self.fc_layers[i](x))
        return x

    def converge_on_batch(
        self,
        examples: jax.Array,
        *,
        optim_x,
        loss,
        T,
    ) -> list[list[jax.Array]]:
        grad_and_values = pxu.grad_and_values(
            px.f(px.NodeParam)(frozen=False),
        )(loss)

        energies: list[list[jax.Array]] = []

        with pxu.eval(self, examples):
            # Run to convergence. Since while loops are hard in JAX, we run for a fixed number of steps.
            # FIXME: XLA compilation takes minutes because unrolling the loop for too many steps is expensive. It takes 6 mins for 625 steps.
            for i in range(T * self.p.T_convergence_multiplier):
                with pxu.step(self):
                    g, _ = grad_and_values(examples, model=self)

                    energies.append([jnp.sum(x.energy()) for x in self.pc_nodes])

                    optim_x(g)
        return energies

    @property
    def num_layers(self) -> int:
        return self.p.num_hidden_layers + 2

    @property
    def prediction(self) -> jax.Array:
        # Return the output ("u" is equal to "x" if the target is not fixed,
        # while it is the actual output of the model if the target is fixed)
        res = self.pc_nodes[-1]["u"]
        assert isinstance(res, jax.Array)
        return res

    @property
    def internal_state(self) -> jax.Array | None:
        res = self.pc_nodes[0]["x"]
        assert res is None or isinstance(res, jax.Array)
        return res

    def x_parameters(self) -> pxc.ParamDict:
        res = self.parameters().filter(px.f(px.NodeParam)(frozen=False))
        assert isinstance(res, pxc.ParamDict)
        return res

    def w_parameters(self) -> pxc.ParamDict:
        res = self.parameters().filter(px.f(px.LayerParam))
        assert isinstance(res, pxc.ParamDict)
        return res

    def save_weights(self, savedir: str) -> None:
        os.makedirs(savedir, exist_ok=True)

        weights = {}
        id_to_name = {}
        for param_name, param_value in self.w_parameters().items():
            param_id = str(uuid4())
            id_to_name[param_id] = param_name
            weights[param_id] = param_value.value

        jnp.savez(os.path.join(savedir, "w_params.npz"), **weights)
        with open(os.path.join(savedir, "w_params_id_to_name.json"), "w") as outfile:
            json.dump(id_to_name, outfile, indent=4)

    def load_weights(self, savedir: str) -> None:
        with open(os.path.join(savedir, "w_params_id_to_name.json"), "r") as infile:
            id_to_name = json.load(infile)

        with np.load(os.path.join(savedir, "w_params.npz")) as npzfile:
            weights: dict[str, jax.Array] = {
                id_to_name[param_id]: jnp.array(npzfile[param_id])
                for param_id in npzfile.files
            }

            missing_parameters = set()
            for param_name, param in self.w_parameters().items():
                if param_name in weights:
                    if param.value.shape != weights[param_name].shape:
                        raise ValueError(
                            f"Parameter {param_name} has shape {param.value.shape} but loaded weight has shape {weights[param_name].shape}"
                        )
                    param.value = weights[param_name]
                else:
                    missing_parameters.add(param_name)

        if missing_parameters:
            logging.error(
                f"When loadings weights {len(missing_parameters)} were not found: {missing_parameters}"
            )


@pxu.vectorize(px.f(px.NodeParam, with_cache=True), in_axis=(0,), out_axis=("sum",))  # type: ignore
def loss(example: jax.Array, *, model: PCDecoder) -> jax.Array:
    model(example=example)
    res = model.energy()
    assert isinstance(res, jax.Array)
    return res


@pxu.vectorize(px.f(px.NodeParam, with_cache=True), in_axis=(0,))
def predict(example: jax.Array, *, model: PCDecoder) -> jax.Array:
    res = model(example=example)
    assert isinstance(res, jax.Array)
    return res


@pxu.vectorize(in_axis=(0,))
def feed_forward_predict(internal_state: jax.Array, *, model: PCDecoder) -> jax.Array:
    """Feed forward predict is not using PCLayers. Instead, it runs internal_state through model weights"""
    res = model.feed_forward_predict(internal_state)
    assert isinstance(res, jax.Array)
    return res


@pxu.jit()
def get_internal_states_on_batch(
    examples, *, model: PCDecoder, optim_x, loss, T
) -> jax.Array:
    model.converge_on_batch(examples, optim_x=optim_x, loss=loss, T=T)
    assert model.internal_state is not None
    return model.internal_state
