__all__ = ["Module", "Layer"]

import jax
import jax.tree_util as jt
from typing import Callable, Dict, Any, Tuple, Union

from ..core import Module as _Module, ModuleList, DEFAULT_GENERATOR
from ..core.util import positional_args_names
from .variables import NodeVar, CachedVar


class Module(_Module):
    @staticmethod
    def _get_submodules(values):
        for v in values:
            if isinstance(v, Module):
                yield v
            elif isinstance(v, ModuleList):
                yield from Module._get_submodules(v)

    def clear(self, *args, **kwargs):
        for m in Module._get_submodules(self.__dict__.values()):
            m.clear(*args, **kwargs)

    @property
    def energy(self):
        return jt.tree_reduce(
            lambda x, y: x + y,
            tuple(m.energy for m in Module._get_submodules(self.__dict__.values())),
        )


def _init_fn(self, rkey):
    self["x"] = self["u"]


def _forward_fn(self, rkey):
    pass


def _energy_fn(x: jax.Array, u: jax.Array, rkey):
    return ((x - u) ** 2).sum()


class Layer(Module):
    def __init__(
            self,
            rkey=DEFAULT_GENERATOR,
            init_fn: Callable[['Layer'], None] = _init_fn,
            forward_fn: Callable[['Layer'], None] = _forward_fn,
            energy_fn: Callable[[Any], jax.Array] = _energy_fn,
            blueprints: Dict[str, Callable[[Any], jax.Array]] = {}
    ):
        super().__init__()

        self.x = NodeVar()
        self.activations = CachedVar()
        self.blueprints = {}

        self.init_fn = init_fn
        self.forward_fn = forward_fn

        self.register_blueprints((("e", energy_fn),))

        self.register_blueprints(blueprints.items())

    def __call__(self, *args, rkey=DEFAULT_GENERATOR, **kwargs):
        arg_names = ("u",) + tuple("u" + str(i) for i in range(1, len(args)))

        for arg_name, arg in zip(arg_names, args):
            self.activations[arg_name] = arg

        for key, value in kwargs.items():
            self.activations[key] = value

        if self.x.value is None:
            self.init_fn(self, rkey)
        else:
            self.forward_fn(self, rkey)

        return self

    def __setitem__(self, key: str, value: jax.Array):
        if key == "x":
            self.x.value = value
        else:
            self.activations[key] = value

    def __getitem__(self, key: Union[str, Tuple[str, Any]]):
        if isinstance(key, tuple):
            key, rkey = key
        else:
            rkey = DEFAULT_GENERATOR

        if key == "x":
            return self.x.value

        if key not in self.activations:
            self.call_blueprint(key, rkey)

        return self.activations[key]

    @property
    def energy(self):
        return self["e"]

    def clear(self, x: bool = False):
        self.activations.clear()
        if x:
            self.x.value = None

    def register_blueprints(self, blueprints: Tuple[str, Callable[[Any], jax.Array]]):
        for key, blueprint in blueprints:
            self.blueprints[key] = blueprint

    def call_blueprint(self, key: str, rkey=DEFAULT_GENERATOR):
        blueprint = self.blueprints[key]

        self.activations[key] = blueprint(
            *(self[k] for k in positional_args_names(blueprint)[:-1]), rkey
        )
