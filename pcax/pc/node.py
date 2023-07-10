__all__ = [
    "Node",
]

import jax
from typing import Callable, Dict, Any, Tuple, Union, Optional

from ..core import RKG, RandomKeyGenerator, ParamCache
from .parameters import NodeParam
from .energymodule import EnergyModule


class VarView:
    def __init__(self, slices: Optional[Union[Tuple[slice], str]] = None) -> None:
        if isinstance(slices, str):
            slices = tuple(
                slice(*tuple(map(lambda i: int(i), s.split(":"))))
                for s in slices.split(",")
            )

        self.slices = slices

    def __getitem__(self, var):
        if self.slices is None:
            return var.value
        return var.value[self.slices]

    def __setitem__(self, var, value):
        if self.slices is None:
            var.value = value
        var.value = var.value.at[self.slices].set(value)


def _init_fn(self, rkg: RandomKeyGenerator):
    self["x"] = self["u"]


def _forward_fn(self, rgk: RandomKeyGenerator):
    pass


def _energy_fn(self, rkg: RandomKeyGenerator):
    e = self["x"] - self["u"]
    return 0.5 * (e * e).sum(axis=-1)


class Node(EnergyModule):
    def __init__(
        self,
        rkg: RandomKeyGenerator = RKG,
        init_fn: Optional[Callable[["Node"], None]] = None,
        forward_fn: Optional[Callable[["Node"], None]] = None,
        energy_fn: Callable[[Any], jax.Array] = _energy_fn,
        blueprints: Dict[str, Callable[[Any], jax.Array]] = {},
        views: Dict[str, VarView] = {},
    ):
        super().__init__()

        self.x = NodeParam()
        self.x_tmp = ParamCache(self.x)
        self.blueprints = {}
        self.views = {
            "u": VarView(),
            **views,
        }

        self.init_fn = init_fn or _init_fn
        self.forward_fn = forward_fn or _forward_fn

        self.register_blueprints((("e", energy_fn),) + tuple(blueprints.items()))

    def __call__(
        self, u: jax.Array = None, rkg: RandomKeyGenerator = RKG, **kwargs
    ):
        if u is not None:
            self.set_activation("u", u)

        for key, value in kwargs.items():
            self.set_activation(key, value)

        if self.is_init:
            self.init_fn(self, rkg)
        else:
            self.forward_fn(self, rkg)

        return self

    def __setitem__(self, key: str, value: jax.Array):
        if key == "x":
            self.x.value = value
        elif key.startswith("x:"):
            self.views[key.split(":", 1)[1]][self.x] = value
        else:
            self.x_tmp[key] = value

    def __getitem__(self, key: Union[str, Tuple[str, Any]]):
        if isinstance(key, tuple):
            key, rkg = key
        else:
            rkg = RKG

        if key == "x":
            return self.x.value
        elif key.startswith("x:"):
            return self.views[key.split(":", 1)[1]][self.x]

        if key not in self.x_tmp:
            self.call_blueprint(key, rkg)

        return self.x_tmp[key]

    def set_activation(self, key: str, value: jax.Array):
        if key in self.x_tmp:
            self.x_tmp[key] = self.x_tmp[key] + value
        else:
            self.x_tmp[key] = value

    def energy(self):
        return self["e"]

    def clear_cache(self):
        self.x_tmp.clear()

    def clear_nodes(self):
        self.x.value = None

    def register_blueprints(self, blueprints: Tuple[str, Callable[[Any], jax.Array]]):
        for key, blueprint in blueprints:
            self.blueprints[key] = blueprint

    def call_blueprint(self, key: str, rkg: RandomKeyGenerator = RKG):
        blueprint = self.blueprints[key]

        self.x_tmp[key] = blueprint(self, rkg)


# def _layerwsigma_init_fn(layer, rkg: RandomKeyGenerator):
#     layer["x"] = layer["u"]

#     if layer.logsigma.value is None:
#         layer.logsigma.value = jax.numpy.zeros(layer["x"].shape)


# def _layerwsigma_energy_fn(layer, rkg: RandomKeyGenerator):
#     return (
#         0.5
#         * (
#             ((layer["x"] - layer["u"]) ** 2 / jax.numpy.exp(layer.logsigma.value))
#             + layer.logsigma.value
#         ).sum()
#     )


# class LayerWSigma(Node):
#     def __init__(
#         self,
#         rkey: RandomKeyGenerator = RKG,
#         init_fn: Callable[["Node"], None] = _layerwsigma_init_fn,
#         forward_fn: Callable[["Node"], None] = _forward_fn,
#         energy_fn: Callable[[Any], jax.Array] = _layerwsigma_energy_fn,
#         blueprints: Dict[str, Callable[[Any], jax.Array]] = {},
#         views: Dict[str, VarView] = {},
#     ):
#         super().__init__(rkey, init_fn, forward_fn, energy_fn, blueprints, views)

#         self.logsigma = Param()
