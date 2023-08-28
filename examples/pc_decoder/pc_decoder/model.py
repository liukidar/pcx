import json
import logging
import os
from typing import Callable
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np
from pc_decoder.params import Params

import pcax as px  # type: ignore
import pcax.core as pxc  # type: ignore
import pcax.nn as nn  # type: ignore
import pcax.utils as pxu  # type: ignore

activation_functions: dict[str | None, Callable[[jax.Array], jax.Array]] = {
    None: lambda x: x,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
    "sigmoid": jax.nn.sigmoid,
}


class PCDecoder(px.EnergyModule):
    def __init__(
        self,
        *,
        params: Params,
        internal_state_init_fn: Callable[
            [Params, jax.random.KeyArray], tuple[jax.Array, jax.random.KeyArray]
        ],
    ) -> None:
        super().__init__()

        self.p = params.copy()
        self.act_fn_hidden = activation_functions[params.activation_hidden]
        self.act_fn_output = activation_functions[params.activation_output]
        self.internal_state_init_fn = internal_state_init_fn
        self.init_prng_key = jax.random.PRNGKey(100)
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
        self.pc_nodes: list[px.Node] = [px.Node() for _ in range(self.num_layers + 1)]

        self.pc_nodes[-1].x.frozen = True

    def __call__(
        self, example: jax.Array | None = None, network_input: jax.Array | None = None
    ) -> jax.Array:
        if network_input is None:
            if self.prior_layer is not None:
                network_input = self.prior_layer(jnp.ones((1,)))
            else:
                if self.pc_nodes[0].is_init:
                    network_input, self.init_prng_key = self.internal_state_init_fn(
                        self.p, self.init_prng_key
                    )
                else:
                    network_input = self.internal_state
        assert network_input is not None
        x = self.pc_nodes[0](network_input)["x"]

        for i in range(self.num_layers):
            act_fn = (
                self.act_fn_hidden if i < self.num_layers - 1 else self.act_fn_output
            )
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
            act_fn = (
                self.act_fn_hidden if i < self.num_layers - 1 else self.act_fn_output
            )
            x = act_fn(self.fc_layers[i](x))
        return x

    def converge_on_batch(
        self,
        examples: jax.Array,
        *,
        optim_x: pxu.Optim,
        loss_fn: Callable,
    ) -> pxu.EnergyMinimizationLoop.LoopState:
        with pxu.eval(self, examples):
            if self.p.reset_optimizer_x_state:
                optim_x.init_state()

            loop = pxu.EnergyMinimizationLoop(
                model=self,
                loss_fn=loss_fn,
                max_iter_number=self.p.T,
                min_iter_number=self.p.T_min_x_updates,
                energy_convergence_threshold=self.p.energy_slow_accurate_convergence_threshold,
                should_update_x=True,
                optim_x=optim_x,
            )
            final_state = loop.run(examples)

        return final_state

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
    def internal_state(self) -> jax.Array:
        res = self.pc_nodes[0]["x"]
        assert isinstance(res, jax.Array)
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
def model_energy_loss(example: jax.Array, *, model: PCDecoder) -> jax.Array:
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
