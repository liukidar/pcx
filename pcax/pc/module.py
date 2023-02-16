__all__ = ["Module", "Layer"]

import jax
import jax.tree_util as jt
from typing import Callable, Dict, Any, Tuple, Union, Optional

from ..core import Module as _Module, ModuleList, DEFAULT_GENERATOR, Generator
from ..core.util import positional_args_names
from .variables import NodeVar, CachedVar


class Module(_Module):
    def __init__(self) -> None:
        super().__init__()

        self.cache = CachedVar()

    @staticmethod
    def _get_submodules(values):
        for v in values:
            if isinstance(v, Module):
                yield v
            elif isinstance(v, ModuleList):
                yield from Module._get_submodules(v)

    def clear_cache(self):
        for m in Module._get_submodules(self.__dict__.values()):
            m.clear_cache()

        self.cache.clear()

    def clear_nodes(self):
        for m in Module._get_submodules(self.__dict__.values()):
            m.clear_nodes()

    def energy(self):
        if "e" not in self.cache:
            self.cache["e"] = jt.tree_reduce(
                lambda x, y: x + y,
                tuple(m.energy() for m in Module._get_submodules(self.__dict__.values())),
            )

        return self.cache["e"]

    def train(self):
        pass

    def eval(self):
        pass


class VarView:
    def __init__(self, slices: Optional[Union[Tuple[slice], str]] = None) -> None:
        if isinstance(slices, str):
            slices = tuple(slice(*tuple(map(lambda i: int(i), s.split(":")))) for s in slices.split(","))

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
    return 0.5 * ((self["x"] - self["u"]) ** 2).sum()


class Layer(Module):
    def __init__(
            self,
            rkey=DEFAULT_GENERATOR,
            init_fn: Callable[['Layer'], None] = _init_fn,
            forward_fn: Callable[['Layer'], None] = _forward_fn,
            energy_fn: Callable[[Any], jax.Array] = _energy_fn,
            blueprints: Dict[str, Callable[[Any], jax.Array]] = {},
            views: Dict[str, VarView] = {}
    ):
        super().__init__()

        self.x = NodeVar()
        self.cache = CachedVar()
        self.blueprints = {}
        self.views = {
            "u": VarView(),
            **views,
        }

        self.init_fn = init_fn
        self.forward_fn = forward_fn

        self.register_blueprints((("e", energy_fn),) + tuple(blueprints.items()))

    def __call__(self, u: jax.Array = None, rkey: Generator = DEFAULT_GENERATOR, **kwargs):
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
            self.cache[key] = value

    def __getitem__(self, key: Union[str, Tuple[str, Any]]):
        if isinstance(key, tuple):
            key, rkey = key
        else:
            rkey = DEFAULT_GENERATOR

        if key == "x":
            return self.x.value
        elif key.startswith("x:"):
            return self.views[key.split(":", 1)[1]][self.x]

        if key not in self.cache:
            self.call_blueprint(key, rkey)

        return self.cache[key]

    def set_activation(self, key: str, value: jax.Array):
        if key in self.cache:
            self.cache[key] = self.cache[key] + value
        else:
            self.cache[key] = value

    def energy(self):
        return self["e"]

    def clear_cache(self):
        self.cache.clear()

    def clear_nodes(self):
        self.x.value = None

    def register_blueprints(self, blueprints: Tuple[str, Callable[[Any], jax.Array]]):
        for key, blueprint in blueprints:
            self.blueprints[key] = blueprint

    def call_blueprint(self, key: str, rkey: Generator = DEFAULT_GENERATOR):
        blueprint = self.blueprints[key]

        self.cache[key] = blueprint(
            self, rkey
        )
