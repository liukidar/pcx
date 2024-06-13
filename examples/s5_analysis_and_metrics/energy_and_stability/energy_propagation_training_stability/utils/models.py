from typing import Callable, List

import jax
import jax.numpy as jnp
import pcax as px
import pcax.predictive_coding as pxc

import utils
from utils.layers import LinearWithCustomInit

cluster = utils.cluster.ClusterManager()


def get_init_func(init_h_sd: float):
    def randn(vode, key, value, rkg):
        return jax.random.normal(rkg(), shape=value.shape) * init_h_sd  # 0.1 is just my guess of a good value

    return randn


class Model(pxc.EnergyModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: int | List[int],
        output_dim: int,
        num_layers: int,
        act_fn: Callable[[jax.Array], jax.Array],
        random_key_generator: px.RandomKeyGenerator = None,
        init_w: str = "default",
        init_h: str = "forward",
        init_h_sd: float = 0.1,
    ) -> None:
        super().__init__()
        assert init_h in ["forward", "randn"], f"Unknown init_h: {init_h}"
        assert init_h_sd > 0, f"init_h_sd should be positive. Got {init_h_sd}"

        self.act_fn = px.static(act_fn)

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * (num_layers - 2)

        dims = [input_dim] + hidden_dims + [output_dim]

        assert len(dims) == num_layers, f"Number of dims should be equal to the number of layers. Got {len(dims)} and {num_layers}."

        self.layers = [
            LinearWithCustomInit(in_dim, out_dim, rkg=random_key_generator, init=init_w) for in_dim, out_dim in zip(dims[:-1], dims[1:])
        ]

        if init_h == "forward":
            vode_kwargs = {}
        elif init_h == "randn":
            vode_kwargs = {
                "ruleset": {pxc.STATUS.INIT: ("h <- u:randn",), "init_inference": ("h <- u",)},
                "tforms": {"randn": get_init_func(init_h_sd)},
            }
        self.vodes = [pxc.Vode((out_dim,), **vode_kwargs) for out_dim in dims[1:-1]] + [pxc.Vode((output_dim,), pxc.ce_energy)]

        self.vodes[-1].h.frozen = True

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray = None) -> jnp.ndarray:
        raise NotImplementedError("This method should be implemented in the subclass.")


class ModelPC(Model):
    def __call__(self, x, y):
        for v, l in zip(self.vodes[:-1], self.layers[:-1]):
            x = v(self.act_fn(l(x)))

        x = self.vodes[-1](self.layers[-1](x))

        if y is not None:
            self.vodes[-1].set("h", y)

        return self.vodes[-1].get("u")


class ModelBP(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # remove earlier vodes, only keep last one
        self.vodes = self.vodes[-1:]

    def __call__(self, x, y):
        for layer in self.layers:
            x = self.act_fn(layer(x))

        x = self.vodes[-1](x)

        if y is not None:
            self.vodes[-1].set("h", y)

        return self.vodes[-1].get("u")
