__all__ = [
    "Module",
    "Function"
]

import abc
from typing import Tuple, Callable, Any

import jax

from .util import repr_function
from .parameters import _BaseParameter, ParamsDict

########################################################################################################################
#
# MODULES
#
########################################################################################################################

# Utils ################################################################################################################


class _ParameterPlaceholder(_BaseParameter):
    """Used internally to represent an empty parameter when flattening/unflattening a module."""
    @property
    def value(self) -> None:
        return None

    @value.setter
    def value(self, _: None):
        return


# Core #################################################################################################################


def flatten_module_with_keys(module: 'Module') -> Tuple[Any, Any]:
    """Flatten a module into a tuple of its parameters and a tuple of its submodules. This is different from how JAX
    flattens a pytree, which is a tuple of leaves and a tree definition. This is because we want to be able to simply
    extract all the parameters of a module to track them through a JAX transformation."""

    parameters = []
    keys = []
    values = []

    for k, v in module.__dict__.items():
        keys.append(k)
        if isinstance(v, _BaseParameter):
            parameters.append((k, v))
            values.append(_ParameterPlaceholder())
        else:
            leaves, treedef = jax.tree_util.tree_flatten_with_path(v, is_leaf=lambda x: isinstance(x, _BaseParameter))
            parameters.extend((f"{k}.{path}", leaf) for path, leaf in leaves if isinstance(leaf, _BaseParameter))
            v = (treedef, tuple(
                (leaf if not isinstance(leaf, _BaseParameter) else _ParameterPlaceholder()) for _, leaf in leaves
            ))
            values.append(v)

    return parameters, (module, keys, values)


def unflatten_module(static_data: Any, parameters: Any) -> 'Module':
    module, keys, values = static_data
    parameters_iter = iter(parameters)
    for key, value in zip(keys, values):
        if isinstance(value, _BaseParameter):
            value = next(parameters_iter)
        else:
            treedef, leaves = value
            value = jax.tree_util.tree_unflatten(treedef, (
                (leaf if not isinstance(leaf, _ParameterPlaceholder) else next(parameters_iter)) for leaf in leaves
            ))
        object.__setattr__(module, key, value if value is not None else next(parameters_iter))

    return module


class _ModuleMeta(abc.ABCMeta):
    """
    Metaclass to register all modules in the JAX pytree flatten/unflatten util.
    A module is flatten by separating its parameters from the rest of the object.
    """

    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        jax.tree_util.register_pytree_with_keys(
            cls, flatten_module_with_keys, unflatten_func=unflatten_module
        )

        return cls


# Modules ##############################################################################################################


class Module(metaclass=_ModuleMeta):
    """A pcax Module is analogous to a Pytorch module: it is the base class each model should inherit from, as it is
    used to keep track of all Parameter used by such model.
    """

    def parameters(self) -> ParamsDict:
        """Returns a dictionary of all the parameters of the module."""

        params = ParamsDict()
        parameters, _ = jax.tree_util.tree_flatten_with_path(self, is_leaf=lambda x: isinstance(x, _BaseParameter))
        scope = f"({self.__class__.__name__})."
        for k, v in parameters:
            params[scope + k[0]] = v

        return params

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Optional module __call__ method, typically a forward pass computation for standard primitives."""
        raise NotImplementedError


class Function(Module):
    """Turn a function into a Module by storing the parameters it uses."""

    def __init__(self, f: Callable, params: ParamsDict):
        """Function constructor.

        Args:
            f: the function or the module to represent.
            params: the ParamsDict of variables used by the function.
        """
        if hasattr(f, "__name__"):
            self.params = ParamsDict((f"{{{f.__name__}}}{k}", v) for k, v in params.items())
        else:
            self.params = ParamsDict(params)
        self.__wrapped__ = f

    def __call__(self, *args, **kwargs):
        """Call the wrapped function."""
        return self.__wrapped__(*args, **kwargs)

    def parameters(self) -> ParamsDict:
        return self.params

    def __repr__(self):
        return f"{self.__class__.__name__}(f={repr_function(self.__wrapped__)})"
