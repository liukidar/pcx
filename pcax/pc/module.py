__all__ = ["Module", "Layer"]

import jax
import jax.tree_util as jt
from typing import Callable, Dict, Any, Tuple, Union, Optional

from ..core import Module as _Module, RKG, RandomKeyGenerator
from .variables import NodeVar
from ..core.parameters import _BaseParameter, Parameter, ParameterCache


class Module(_Module):
    def __init__(self) -> None:
        super().__init__()

    def clear_cache(self):
        parameters = jax.tree_util.tree_leaves(self, is_leaf=lambda x: isinstance(x, _BaseParameter))
        for p in parameters:
            if isinstance(p, ParameterCache):
                p.clear()

    def clear_nodes(self):
        parameters = jax.tree_util.tree_leaves(self, is_leaf=lambda x: isinstance(x, _BaseParameter))
        for p in parameters:
            if isinstance(p, NodeVar):
                p.value = None

    def energy(self):
        modules = tuple(
            m for m in jax.tree_util.tree_leaves(
                tuple(self.__dict__.values()),
                is_leaf=(lambda x: isinstance(x, Module))
            ) if isinstance(m, Module)
        )
        return jt.tree_reduce(
            lambda x, y: x + y,
            tuple(
                m.energy()
                for m in modules
            ),
        )

    def train(self):
        pass

    def eval(self):
        pass


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


def _init_fn(self, rkey):
    self["x"] = self["u"]


def _forward_fn(self, rkey):
    pass


def _energy_fn(self, rkey):
    e = self["x"] - self["u"]
    return 0.5 * (e * e).sum(axis=-1)


class Layer(Module):
    def __init__(
        self,
        rkey=RKG,
        init_fn: Callable[["Layer"], None] = _init_fn,
        forward_fn: Callable[["Layer"], None] = _forward_fn,
        energy_fn: Callable[[Any], jax.Array] = _energy_fn,
        blueprints: Dict[str, Callable[[Any], jax.Array]] = {},
        views: Dict[str, VarView] = {},
    ):
        super().__init__()

        self.x = NodeVar()
        self.x_tmp = ParameterCache(self.x)
        self.blueprints = {}
        self.views = {
            "u": VarView(),
            **views,
        }

        self.init_fn = init_fn
        self.forward_fn = forward_fn

        self.register_blueprints((("e", energy_fn),) + tuple(blueprints.items()))

    def __call__(
        self, u: jax.Array = None, rkey: RandomKeyGenerator = RKG, **kwargs
    ):
        if u is not None:
            self.set_activation("u", u)

        for key, value in kwargs.items():
            if key not in self.views:
                raise ValueError(f"Unregistered key {key}")
            else:
                self.set_activation(key, value)

        if self.x.value is None:
            self.init_fn(self, rkey)
        else:
            self.forward_fn(self, rkey)

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
            key, rkey = key
        else:
            rkey = RKG

        if key == "x":
            return self.x.value
        elif key.startswith("x:"):
            return self.views[key.split(":", 1)[1]][self.x]

        if key not in self.x_tmp:
            self.call_blueprint(key, rkey)

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

    def call_blueprint(self, key: str, rkey: RandomKeyGenerator = RKG):
        blueprint = self.blueprints[key]

        self.x_tmp[key] = blueprint(self, rkey)


def _layerwsigma_init_fn(layer, rkey):
    layer["x"] = layer["u"]

    if layer.logsigma.value is None:
        layer.logsigma.value = jax.numpy.zeros(layer["x"].shape)


def _layerwsigma_energy_fn(layer, rkey):
    return (
        0.5
        * (
            ((layer["x"] - layer["u"]) ** 2 / jax.numpy.exp(layer.logsigma.value))
            + layer.logsigma.value
        ).sum()
    )


class LayerWSigma(Layer):
    def __init__(
        self,
        rkey=RKG,
        init_fn: Callable[["Layer"], None] = _layerwsigma_init_fn,
        forward_fn: Callable[["Layer"], None] = _forward_fn,
        energy_fn: Callable[[Any], jax.Array] = _layerwsigma_energy_fn,
        blueprints: Dict[str, Callable[[Any], jax.Array]] = {},
        views: Dict[str, VarView] = {},
    ):
        super().__init__(rkey, init_fn, forward_fn, energy_fn, blueprints, views)

        self.logsigma = Parameter()
