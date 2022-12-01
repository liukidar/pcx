__all__ = ["Module", "Layer"]

from ..core import Module as _Module, ModuleList
from ..core.util import positional_args_names
from .variables import NodeVar, CachedVar

import jax.tree_util as jt


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


class Layer(Module):
    def __init__(self):
        super().__init__()

        self.x = NodeVar()
        self.activations = CachedVar()
        self.blueprints = {}

        def init_fn(self):
            self["x"] = self["u"]

        self.init_fn = init_fn
        self.forward_fn = lambda self: None

        self.register_blueprint("e", lambda x, u: ((x - u) ** 2).sum())

    def __call__(self, *args, **kwargs):
        arg_names = ("u",) + tuple("u" + str(i) for i in range(1, len(args)))

        for arg_name, arg in zip(arg_names, args):
            self.activations[arg_name] = arg

        for key, value in kwargs.items():
            self.activations[key] = value

        if self.x.value is None:
            self.init_fn(self)
        else:
            self.forward_fn(self)

        return self

    def __setitem__(self, key, value):
        if key == "x":
            self.x.value = value
        else:
            self.activations[key] = value

    def __getitem__(self, key):
        if key == "x":
            return self.x.value

        if key not in self.activations:
            self.call_blueprint(key)

        return self.activations[key]

    @property
    def energy(self):
        return self["e"]

    def clear(self, x: bool = False):
        self.activations.clear()
        if x:
            self.x.value = None

    def register_blueprint(self, key, blueprint):
        self.blueprints[key] = blueprint

    def call_blueprint(self, key):
        blueprint = self.blueprints[key]

        self.activations[key] = blueprint(
            *(self[k] for k in positional_args_names(blueprint))
        )
