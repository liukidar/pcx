__all__ = [
    "Module",
    "Function",
    "to_stateless",
    "to_stateful",
    "pure_fn"
]

import abc
from typing import Tuple, Callable, Any, Optional
import functools

import jax
import equinox as eqx

from .util import repr_function
from .parameters import _AbstractParam, ParamDict

########################################################################################################################
#
# MODULES
#
########################################################################################################################


# Core #################################################################################################################


def flatten_module_with_keys(module: 'Module') -> Tuple[Any, Any]:
    return tuple(module.__dict__.items()), tuple(module.__dict__.keys())


def unflatten_module(static_data: Any, parameters: Any, *, cls) -> 'Module':
    module = object.__new__(cls)

    for k, v in zip(static_data, parameters):
        object.__setattr__(module, k, v)

    return module


class _ModuleMeta(abc.ABCMeta):
    """
    Metaclass to register all modules in the JAX pytree flatten/unflatten util.
    A module is flatten by separating its parameters from the rest of the object.
    """

    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        jax.tree_util.register_pytree_with_keys(
            cls, flatten_module_with_keys, unflatten_func=functools.partial(unflatten_module, cls=cls)
        )

        return cls


def to_stateless(pytree: Any, keep_values: bool = False):
    # Replace all the model parameters with their values
    params, pytree = eqx.partition(
        pytree,
        lambda x: isinstance(x, _AbstractParam),
        is_leaf=lambda x: isinstance(x, _AbstractParam)
    )

    # Copy the raw tensor values in the new pytree (replacing the None values)
    if keep_values is True:
        pytree = jax.tree_util.tree_map(
            lambda s, p: p.value if isinstance(p, _AbstractParam) else s,
            pytree,
            params,
            is_leaf=lambda x: x is None
        )

    return pytree, params


def to_stateful(pytree: Any, params: Any, keep_values: bool = False):
    if keep_values is True:
        # Set params values to the pytree's values
        def s2p(p, s):
            p.value = s

            return p

        params = jax.tree_util.tree_map(
            s2p,
            params,
            pytree,
            is_leaf=lambda x: isinstance(x, _AbstractParam)
        )

    # Combine pytree and params
    pytree = jax.tree_util.tree_map(
        lambda p, s: p if isinstance(p, _AbstractParam) else s,
        params,
        pytree,
        is_leaf=lambda x: isinstance(x, _AbstractParam) or x is None
    )

    return pytree


def pure_fn(fn):
    """Transform a function into a pure args function. A pure function sees all pcax.Params as jax.Array.
    NOTE: being pure, all changes to args parameters that are not Modules are not tracked."""
    @functools.wraps(fn)
    def wrap_fn(*args, **kwargs):
        (args, kwargs), params = to_stateless((args, kwargs), keep_values=True)

        r = fn(*args, **kwargs)

        (args, kwargs) = to_stateful((args, kwargs), params, keep_values=True)

        return r

    return wrap_fn

# Modules ##############################################################################################################


class Module(metaclass=_ModuleMeta):
    """A pcax Module is analogous to a Pytorch module: it is the base class each model should inherit from, as it is
    used to keep track of all Parameters used by such model.
    """

    def __init__(self) -> None:
        self._mode = None

    def parameters(self) -> ParamDict:
        """Returns a dictionary of all the parameters of the module."""

        _, params = to_stateless(self)

        return ParamDict.from_pytree(params, f"({self.__class__.__name__}).")

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Optional module __call__ method, typically a forward pass computation for standard primitives."""
        raise NotImplementedError

    def set_mode(self, mode: str):
        self._mode = mode
        for m in self.get_submodules():
            m.set_mode(mode)

    def train(self):
        self.set_mode("train")

    def eval(self):
        self.set_mode("eval")

    @property
    def is_train(self):
        return self._mode == "train"

    @property
    def is_eval(self):
        return self._mode == "eval"

    def get_submodules(self, *, cls: Optional[type] = None):
        cls = cls or Module

        for v in self.__dict__.values():
            for leaf in jax.tree_util.tree_leaves(v, is_leaf=lambda x: isinstance(x, Module)):
                if isinstance(leaf, cls):
                    yield leaf


class Function(Module):
    """Turn a function into a Module by storing the parameters it uses."""

    def __init__(self, f: Callable, params: ParamDict):
        """Function constructor.

        Args:
            f: the function or the module to represent.
            params: the ParamsDict of variables used by the function.
        """
        if hasattr(f, "__name__"):
            self.params = ParamDict((f"{{{f.__name__}}}{k}", v) for k, v in params.items())
        else:
            self.params = ParamDict(params)
        self.__wrapped__ = f

    def __call__(self, *args, **kwargs):
        """Call the wrapped function."""
        return self.__wrapped__(*args, **kwargs)

    def parameters(self) -> ParamDict:
        return self.params

    def __repr__(self):
        return f"{self.__class__.__name__}(f={repr_function(self.__wrapped__)})"
