__all__ = ["BaseParam", "Param", "ParamDict", "ParamCache", "get", "set"]


import abc
from typing import Tuple, Dict, Any, Type
import functools

import jax


########################################################################################################################
#
# PARAMETER
#
# pcax is inspired by objax (https://github.com/google/objax) and equinox (https://github.com/patrick-kidger/equinox),
# two other JAX libraries. The core idea is that each JAX array is wrapped in a Param object that pcax can track through
# JAX transformations, without the need to respect the strict functional programming paradigm of JAX.
#
########################################################################################################################

# Core #################################################################################################################


class _BaseParamMeta(abc.ABCMeta):
    """
    Metaclass to register all parameters in the JAX pytree flatten/unflatten util.
    A parameter is flattened by separating its '_value' from its other attributes, which are considered static.
    """

    def __new__(mcs, name, bases, dct):
        _cls = super().__new__(mcs, name, bases, dct)

        jax.tree_util.register_pytree_with_keys(
            _cls,
            flatten_func=_BaseParamMeta.flatten_parameter,
            flatten_with_keys=_BaseParamMeta.flatten_parameter_with_keys,
            unflatten_func=functools.partial(_BaseParamMeta.unflatten_parameter, cls=_cls),
        )

        return _cls

    @staticmethod
    def flatten_parameter(param: "BaseParam") -> Tuple[Any, Dict[str, Any]]:
        _aux_data = dict.copy(param.__dict__)
        del _aux_data["_value"]

        return (param._value,), _aux_data

    @staticmethod
    def flatten_parameter_with_keys(param: "BaseParam") -> Tuple[Any, Dict[str, Any]]:
        _aux_data = dict.copy(param.__dict__)
        del _aux_data["_value"]

        return ((jax.tree_util.GetAttrKey("value"), param._value),), _aux_data

    @staticmethod
    def unflatten_parameter(aux_data: Dict[str, Any], children: Any, *, cls: Type["BaseParam"]) -> "BaseParam":
        _param = object.__new__(cls)

        _param.__dict__ = dict.copy(aux_data)
        _param._value = children[0]

        return _param


class BaseParam(metaclass=_BaseParamMeta):
    """
    Base abstract class for all parameters. It is used to detect whether an object is a parameter or not.
    """

    def __init__(self, value: jax.Array | Any | None = None):
        """
        _BaseParam constructor.

        Args:
            value: the value (usually a tensor) to wrap. pcax will treat (only!) such values as dynamic.
        """
        self._value = value

    @abc.abstractmethod
    def get(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def set(self, value):
        raise NotImplementedError()

    def __bool__(self):
        raise TypeError(
            "To prevent accidental errors parameters can not be used as Python bool. "
            "To check if variable is `None` use `is None` or `is not None` instead."
        )


# Parameter ###########################################################################################################


class DynamicParam(BaseParam):
    pass


class Param(DynamicParam):
    """
    The base class to represent and store a dynamic value in pcax.
    This is mainly used to wrap jax.Arrays and track them through JAX transformations.
    """

    def get(self) -> jax.Array:
        return self._value

    def set(self, value: jax.Array) -> "Param":
        self._value = value

        return self

    def __repr__(self):
        rvalue = (
            f"[{','.join(map(str, self.shape))}], {self.dtype}"
            if isinstance(self._value, jax.Array)
            else repr(self._value)
        )
        t = f"{self.__class__.__name__}({rvalue})"

        return t

    # Python looks up special methods only on classes, not instances. This means
    # these methods needs to be defined explicitly rather than relying on
    # __getattr__.
    def __neg__(self):
        return self._value.__neg__()  # noqa: E704

    def __pos__(self):
        return self._value.__pos__()  # noqa: E704

    def __abs__(self):
        return self._value.__abs__()  # noqa: E704

    def __invert__(self):
        return self._value.__invert__()  # noqa: E704

    def __eq__(self, __other):
        return self._value.__eq__(get(__other))  # noqa: E704

    def __ne__(self, __other):
        return self._value.__ne__(get(__other))  # noqa: E704

    def __lt__(self, __other):
        return self._value.__lt__(get(__other))  # noqa: E704

    def __le__(self, __other):
        return self._value.__le__(get(__other))  # noqa: E704

    def __gt__(self, __other):
        return self._value.__gt__(get(__other))  # noqa: E704

    def __ge__(self, __other):
        return self._value.__ge__(get(__other))  # noqa: E704

    def __add__(self, __other):
        return self._value.__add__(get(__other))  # noqa: E704

    def __radd__(self, __other):
        return self._value.__radd__(get(__other))  # noqa: E704

    def __iadd__(self, __other):
        self._value = self._value.__add__(get(__other))  # noqa: E704
        return self

    def __sub__(self, __other):
        return self._value.__sub__(get(__other))  # noqa: E704

    def __rsub__(self, __other):
        return self._value.__rsub__(get(__other))  # noqa: E704

    def __isub__(self, __other):
        self._value = self._value.__sub__(get(__other))  # noqa: E704
        return self

    def __mul__(self, __other):
        return self._value.__mul__(get(__other))  # noqa: E704

    def __rmul__(self, __other):
        return self._value.__rmul__(get(__other))  # noqa: E704

    def __imul__(self, __other):
        self._value = self._value.__mul__(get(__other))  # noqa: E704
        return self

    def __div__(self, __other):
        return self._value.__div__(get(__other))  # noqa: E704

    def __rdiv__(self, __other):
        return self._value.__rdiv__(get(__other))  # noqa: E704

    def __idiv__(self, __other):
        self._value = self._value.__div__(get(__other))  # noqa: E704
        return self

    def __truediv__(self, __other):
        return self._value.__truediv__(get(__other))  # noqa: E704

    def __rtruediv__(self, __other):
        return self._value.__rtruediv__(get(__other))  # noqa: E704

    def __floordiv__(self, __other):
        return self._value.__floordiv__(get(__other))  # noqa: E704

    def __rfloordiv__(self, __other):
        return self._value.__rfloordiv__(get(__other))  # noqa: E704

    def __divmod__(self, __other):
        return self._value.__divmod__(get(__other))  # noqa: E704

    def __rdivmod__(self, __other):
        return self._value.__rdivmod__(get(__other))  # noqa: E704

    def __mod__(self, __other):
        return self._value.__mod__(get(__other))  # noqa: E704

    def __rmod__(self, __other):
        return self._value.__rmod__(get(__other))  # noqa: E704

    def __pow__(self, __other):
        return self._value.__pow__(get(__other))  # noqa: E704

    def __rpow__(self, __other):
        return self._value.__rpow__(get(__other))  # noqa: E704

    def __matmul__(self, __other):
        return self._value.__matmul__(get(__other))  # noqa: E704

    def __rmatmul__(self, __other):
        return self._value.__rmatmul__(get(__other))  # noqa: E704

    def __and__(self, __other):
        return self._value.__and__(get(__other))  # noqa: E704

    def __rand__(self, __other):
        return self._value.__rand__(get(__other))  # noqa: E704

    def __or__(self, __other):
        return self._value.__or__(get(__other))  # noqa: E704

    def __ror__(self, __other):
        return self._value.__ror__(get(__other))  # noqa: E704

    def __xor__(self, __other):
        return self._value.__xor__(get(__other))  # noqa: E704

    def __rxor__(self, __other):
        return self._value.__rxor__(get(__other))  # noqa: E704

    def __lshift__(self, __other):
        return self._value.__lshift__(get(__other))  # noqa: E704

    def __rlshift__(self, __other):
        return self._value.__rlshift__(get(__other))  # noqa: E704

    def __rshift__(self, __other):
        return self._value.__rshift__(get(__other))  # noqa: E704

    def __rrshift__(self, __other):
        return self._value.__rrshift__(get(__other))  # noqa: E704

    def __round__(self, ndigits=None):
        return self._value.__round__(ndigits)  # noqa: E704

    def __getitem__(self, __idx):
        return self._value.__getitem__(__idx)

    def __array__(self, dtype=None):
        return self._value.__array__(dtype)

    def __getattr__(self, __name):
        return getattr(self._value, __name)

    @property
    def dtype(self):
        """Wrapped value data type."""
        return self._value.dtype

    @property
    def shape(self):
        """Wrapped value shape."""
        return self._value.shape

    @property
    def ndim(self):
        """Number of dimentions of wrapped value."""
        return self._value.ndim


class ParamDict(DynamicParam):
    def __init__(self, value: Dict[str, jax.Array | Any | None] = None):
        super().__init__(value)

    def __getitem__(self, __key: str) -> Any:
        return self._value[__key]

    def __setitem__(self, __key: str, __value: jax.Array) -> None:
        # Clearing a parameter equates setting its _value to None,
        # so we need to reset it to an empty dictionary when necessary.
        if self._value is None:
            self._value = {}
        self._value[__key] = __value

    def __contains__(self, __key: str) -> bool:
        return __key in self._value

    def get(self, key: str | None = None, default: jax.Array | Any | None = None) -> Any:
        return self._value.get(key, default) if key is not None else self._value

    def set(self, value) -> None:
        self._value = value

    def __repr__(self):
        return f"{self.__class__.__name__}(params={repr(self._value)})"


class ParamCache:
    """
    A simple sentinel class used to identify all parameters used as a temporary cache.
    """

    pass


# Utils ################################################################################################################


def get(x: Any | BaseParam) -> Any:
    """Return the value encapsulated in the input argument if it is a BaseParam, otherwise return the input argument
    itself. Used in ambiguous situations to ensure that the input is a value and not a BaseParam.

    Args:
        x (Any | BaseParam): input argument

    Returns:
        Any: value encapsulated in the input argument if it is a BaseParam, otherwise the input argument itself.
    """
    if isinstance(x, BaseParam):
        return x.get()
    else:
        return x


def set(obj: Any, x: Any | BaseParam) -> Any | BaseParam:
    """Set the value of the input object and returns it if it is a BaseParam, otherwise return the new value itself.
    Used in ambiguous situations to ensure that the input object is correctly updated.

    Returns:
        Any | BaseParam: the updated input object if it is a BaseParam, otherwise the new value itself.
    """
    if isinstance(obj, BaseParam):
        obj.set(get(x))
    else:
        obj = set(x)

    return obj
