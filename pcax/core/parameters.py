__all__ = [
    "get_param",
    "set_param",
    "Param",
    "Param",
    "ParamRef",
    "ParamDict",
    "ParamCache",
]

import abc
import re
from typing import Union, Tuple, Optional, Iterable, Dict, Iterator, Callable, Any
import copy

import jax

from .util import repr_function, move

########################################################################################################################
#
# PARAMETERS
#
# pcax is inspired by objax (https://github.com/google/objax) and equinox (https://github.com/patrick-kidger/equinox),
# two other JAX libraries. The core idea is that each JAX array is wrapped in a _BaseParameter object that allows pcax
# to automatically keep track of it, without the need to respect the strict functional programming paradigm of JAX.
#
########################################################################################################################

# Utils ################################################################################################################


def get_param(x: Union[jax.Array, "Param"]):
    """Returns JAX value encapsulated in the input argument."""
    if isinstance(x, Param):
        return x.value
    else:
        return x


def set_param(obj: Any, x: Union[jax.Array, "Param"]):
    if isinstance(obj, Param):
        obj.value = get_param(x)
    else:
        obj = get_param(x)

    return obj


def reduce_none(x: Optional[jax.Array]) -> jax.Array:
    """Reduces a vmapped value to its first element."""
    return x[0] if x is not None else None


def reduce_mean(x: jax.Array) -> jax.Array:
    """Reduces a vmapped value to the mean of its elements."""
    return x.mean(axis=0)


def reduce_id(x: jax.Array) -> jax.Array:
    """Does not apply any reduction to a vmapped element."""
    return x


# Core #################################################################################################################


class _FlattenAbstractParam:
    """A class to flatten/unflatten a parameter."""

    def __init__(self, param: '_AbstractParam'):
        self.param = param

    def __hash__(self) -> int:
        return self.param.__hash__()

    def __eq__(self, __other: '_FlattenAbstractParam') -> bool:
        return self.param is __other.param

    def __ne__(self, __other: '_FlattenAbstractParam') -> bool:
        return self.param is not __other.param


class _AbstractParamMeta(abc.ABCMeta):
    """
    Metaclass to register all parameters in the JAX pytree flatten/unflatten util.
    A parameter is flatten by separating its value from the rest of the object.
    A parameter is considered the static part of itself; this ensures that a parameter is never destroyed when
    flattening/unflattening it. Thus, the references to a parameter are always valid until the parameter is explicitly
    deleted.
    """

    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        jax.tree_util.register_pytree_with_keys(
            cls,
            flatten_func=_AbstractParamMeta.flatten_parameter,
            flatten_with_keys=_AbstractParamMeta.flatten_parameter_with_keys,
            unflatten_func=_AbstractParamMeta.unflatten_parameter
        )

        return cls

    @staticmethod
    def flatten_parameter(param: '_AbstractParam') -> Tuple[Any, '_FlattenAbstractParam']:
        return (param.value,), _FlattenAbstractParam(param)

    @staticmethod
    def flatten_parameter_with_keys(param: '_AbstractParam') -> Tuple[Any, '_FlattenAbstractParam']:
        return (("value", param.value),), _FlattenAbstractParam(param)

    @staticmethod
    def unflatten_parameter(flatten_param: '_FlattenAbstractParam', data: Any) -> '_AbstractParam':
        param = flatten_param.param
        param.value = data[0]
        return param


class _AbstractParam(metaclass=_AbstractParamMeta):
    """
    Base abstract class for all parameters. It is used to detect whether an object is a parameter or not.
    """

    @property
    @abc.abstractmethod
    def value(self) -> jax.Array:
        raise NotImplementedError("Pure method")

    @value.setter
    @abc.abstractmethod
    def value(self, tensor: jax.Array):
        raise NotImplementedError("Pure method")

    def __bool__(self):
        raise TypeError(
            "To prevent accidental errors parameters can not be used as Python bool. "
            "To check if variable is `None` use `is None` or `is not None` instead."
        )

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, __value: object) -> bool:
        return self is __value

    @abc.abstractmethod
    def reduce(self):
        """Method called by Vectorize to reduce a multiple-device (or batched in case of vectorization)
        value to a single element."""
        raise NotImplementedError("Pure method")


# Parameters ##########################################################################################################


class Param(_AbstractParam):
    """The abstract base class to represent and store a jax.Array."""

    def __init__(
        self,
        value: Optional[jax.Array] = None,
        reduce: Optional[Callable[[jax.Array], jax.Array]] = reduce_none
    ):
        self._value = value
        self._reduce = reduce

    def __move__(self, __other: Optional['Param'] = None) -> 'Param':
        if __other is None:
            __other = copy.copy(self)
        else:
            __other.value = self.value
        self.value = None

        return __other

    @property
    def value(self) -> jax.Array:
        return self._value

    @value.setter
    def value(self, __value: jax.Array):
        self._value = __value

    def reduce(self):
        self.value = self._reduce(self.value)

    def __repr__(self):
        rvalue = re.sub("[\n]+", "\n", repr(self.value))
        t = f"{self.__class__.__name__}({rvalue})"
        if not self._reduce:
            return t
        return f"{t[:-1]}, reduce={repr_function(self._reduce)})"

    # Python looks up special methods only on classes, not instances. This means
    # these methods needs to be defined explicitly rather than relying on
    # __getattr__.
    def __neg__(self):
        return self.value.__neg__()  # noqa: E704

    def __pos__(self):
        return self.value.__pos__()  # noqa: E704

    def __abs__(self):
        return self.value.__abs__()  # noqa: E704

    def __invert__(self):
        return self.value.__invert__()  # noqa: E704

    def __eq__(self, __other):
        return self.value.__eq__(get_param(__other))  # noqa: E704

    def __ne__(self, __other):
        return self.value.__ne__(get_param(__other))  # noqa: E704

    def __lt__(self, __other):
        return self.value.__lt__(get_param(__other))  # noqa: E704

    def __le__(self, __other):
        return self.value.__le__(get_param(__other))  # noqa: E704

    def __gt__(self, __other):
        return self.value.__gt__(get_param(__other))  # noqa: E704

    def __ge__(self, __other):
        return self.value.__ge__(get_param(__other))  # noqa: E704

    def __add__(self, __other):
        return self.value.__add__(get_param(__other))  # noqa: E704

    def __radd__(self, __other):
        return self.value.__radd__(get_param(__other))  # noqa: E704

    def __sub__(self, __other):
        return self.value.__sub__(get_param(__other))  # noqa: E704

    def __rsub__(self, __other):
        return self.value.__rsub__(get_param(__other))  # noqa: E704

    def __mul__(self, __other):
        return self.value.__mul__(get_param(__other))  # noqa: E704

    def __rmul__(self, __other):
        return self.value.__rmul__(get_param(__other))  # noqa: E704

    def __div__(self, __other):
        return self.value.__div__(get_param(__other))  # noqa: E704

    def __rdiv__(self, __other):
        return self.value.__rdiv__(get_param(__other))  # noqa: E704

    def __truediv__(self, __other):
        return self.value.__truediv__(get_param(__other))  # noqa: E704

    def __rtruediv__(self, __other):
        return self.value.__rtruediv__(get_param(__other))  # noqa: E704

    def __floordiv__(self, __other):
        return self.value.__floordiv__(get_param(__other))  # noqa: E704

    def __rfloordiv__(self, __other):
        return self.value.__rfloordiv__(get_param(__other))  # noqa: E704

    def __divmod__(self, __other):
        return self.value.__divmod__(get_param(__other))  # noqa: E704

    def __rdivmod__(self, __other):
        return self.value.__rdivmod__(get_param(__other))  # noqa: E704

    def __mod__(self, __other):
        return self.value.__mod__(get_param(__other))  # noqa: E704

    def __rmod__(self, __other):
        return self.value.__rmod__(get_param(__other))  # noqa: E704

    def __pow__(self, __other):
        return self.value.__pow__(get_param(__other))  # noqa: E704

    def __rpow__(self, __other):
        return self.value.__rpow__(get_param(__other))  # noqa: E704

    def __matmul__(self, __other):
        return self.value.__matmul__(get_param(__other))  # noqa: E704

    def __rmatmul__(self, __other):
        return self.value.__rmatmul__(get_param(__other))  # noqa: E704

    def __and__(self, __other):
        return self.value.__and__(get_param(__other))  # noqa: E704

    def __rand__(self, __other):
        return self.value.__rand__(get_param(__other))  # noqa: E704

    def __or__(self, __other):
        return self.value.__or__(get_param(__other))  # noqa: E704

    def __ror__(self, __other):
        return self.value.__ror__(get_param(__other))  # noqa: E704

    def __xor__(self, __other):
        return self.value.__xor__(get_param(__other))  # noqa: E704

    def __rxor__(self, __other):
        return self.value.__rxor__(get_param(__other))  # noqa: E704

    def __lshift__(self, __other):
        return self.value.__lshift__(get_param(__other))  # noqa: E704

    def __rlshift__(self, __other):
        return self.value.__rlshift__(get_param(__other))  # noqa: E704

    def __rshift__(self, __other):
        return self.value.__rshift__(get_param(__other))  # noqa: E704

    def __rrshift__(self, __other):
        return self.value.__rrshift__(get_param(__other))  # noqa: E704

    def __round__(self, ndigits=None):
        return self.value.__round__(ndigits)  # noqa: E704

    def __getitem__(self, __idx):
        return self.value.__getitem__(__idx)

    def __array__(self, dtype=None):
        return self.value.__array__(dtype)

    def __bool__(self):
        raise TypeError(
            "To prevent accidental errors variables can not be used as Python bool. "
            "To check if variable is `None` use `is None` or `is not None` instead."
        )

    def __getattr__(self, __name):
        # Avoid infinite recursion when copying
        if __name == "_value":
            return None
        return getattr(self._value, __name)

    @property
    def dtype(self):
        """Variable data type."""
        return self.value.dtype

    @property
    def shape(self):
        """Variable shape."""
        return self.value.shape

    @property
    def ndim(self):
        """Number of dimentions."""
        return self.value.ndim


class ParamRef(_AbstractParam):
    """A state parameter that references a trainable parameter for assignment.

    ParamRef are used by optimizers to keep references to trainable variables. This is necessary to differentiate
    them from the optimizer own training variables if any."""

    def __init__(self, param: Param):
        """ParamRef constructor.

        Args:
            param: the Parameter to keep the reference of.
        """
        super().__init__()
        self.ref = param

    def __move__(self, __target: Optional['ParamRef'] = None) -> 'ParamRef':
        if __target is None:
            new_var = copy.copy(self)
        else:
            new_var = __target
            new_var.ref = self.ref
            new_var.value = self.value
        self.ref = None
        self.value = None

        return new_var

    @property
    def value(self) -> jax.Array:
        """To be consistent with other state parameters, ParameterRef have a value property that returns None,
        as it does not store any value to be directly optimized."""
        return None

    @value.setter
    def value(self, _: jax.Array):
        return

    def reduce(self):
        """Similarly to .value, ParameterRef does not apply any reduction to its content."""
        return

    def __repr__(self):
        return f"{self.__class__.__name__}(ref={repr(self.ref)})"


class ParamCache(Param, ParamRef):
    """A parameter dictionary used to store interemediate transformations of the referenced parameter. Those are
    normally only stored in the computation tree and are not accessible to the user."""

    def __init__(
        self, param: Param
    ):
        """ParamCache constructor.

        Args:
            param: the Parameter to keep the reference of.
        """

        Param.__init__(self, {}, None)
        ParamRef.__init__(self, param)

    def __getitem__(self, __key: str):
        return self._value[__key]

    def __setitem__(self, __key: str, __value: jax.Array):
        self._value[__key] = __value

    def __contains__(self, __key: str):
        return __key in self._value

    def get(self, __key: str, __default: Optional[jax.Array] = None):
        return self._value.get(__key, __default)

    def clear(self):
        self._value = {}

    def reduce(self):
        for k, v in self.value.items():
            self.value[k] = self.ref._reduce(v)

    def __repr__(self):
        rvalue = re.sub("[\n]+", "\n", repr(self.value))
        t = f"{self.__class__.__name__}({rvalue})"
        return f"{t[:-1]}"


class ParamDict(Dict[str, _AbstractParam]):
    """A ParamDict is a dictionary (name, var) with some additional methods to make manipulation of collections of
    parameters easy. In particular, iterating through a ParamDict will iterate through each unique parameter stored
    in the dictionary, avoiding duplicate references."""

    @staticmethod
    def from_pytree(pytree, prefix: str = ""):
        leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(
            pytree,
            is_leaf=lambda x: isinstance(x, _AbstractParam)
        )

        return ParamDict(
            map(
                lambda leaf_with_path: (
                    prefix + ".".join(map(lambda k: str(k), leaf_with_path[0])), leaf_with_path[1]
                ),
                leaves_with_path
            )
        )

    def __init__(self, *args, **kwargs):
        """ParamDict constructor.

        Simply initialise the dictionary with the given elements."""
        super().__init__(*args, **kwargs)

    def __move__(self, __target: Optional['ParamDict' | Dict[str, _AbstractParam]] = None):
        params = __target or ParamDict()

        seen = {}
        for k, v in self.items():
            uid = id(v)
            if uid not in seen:
                seen[uid] = move(v, params[k] if __target else None)

            params[k] = seen[uid]

        return params

    def __add__(self, __other: "ParamDict") -> "ParamDict":
        """Overloaded add operator to compute the set union of two ParamDicts."""
        params = ParamDict(self)
        params.update(__other)
        return params

    def __sub__(self, __other: "ParamDict") -> "ParamDict":
        """Overloaded sub operator to compute the set difference between two ParamDicts."""
        params = ParamDict()
        other_ids = set(map(lambda x: id(x), __other.values()))

        for k, v in self.items():
            if id(v) not in other_ids:
                params[k] = v

        return params

    def __iter__(self) -> Iterator[_AbstractParam]:
        """Create an iterator that iterates over the parameters (dict values) and visit them only once.
        If a parameters has two names, for example in the case of weight sharing, this iterator yields the variable only
        once."""
        seen = set()
        for v in self.values():
            uid = id(v)
            if uid not in seen:
                seen.add(uid)
                yield v

    def __setitem__(self, __key: str, __value: _AbstractParam):
        """Overload bracket assignment to catch potential conflicts during assignment."""
        if __key in self and self[__key] is not __value:
            raise ValueError("Name conflicts when appending to ParamsDict", __key)
        dict.__setitem__(self, __key, __value)

    def update(self, other: Union["ParamDict", Iterable[Tuple[str, _AbstractParam]]]):
        """Overload dict.update method to catch potential conflicts during assignment."""
        if not isinstance(other, self.__class__):
            other = list(other)
        else:
            other = other.items()
        for k, v in other:
            if k in self:
                if self[k] is not v:
                    raise ValueError(
                        f"Name conflict when combining ParamsDict {k}"
                    )
            else:
                self[k] = v

    def __str__(self, max_width=100):
        """Pretty print the contents of the ParamsDict."""
        text = []
        count = 0
        longest_string = max((len(x) for x in self.keys()), default=20)
        longest_string = min(max_width, max(longest_string, 20))
        for name, v in self.items():
            count += 1
            text.append(f"{name:{longest_string}}")
        text.append(f'{f"+Total({count})":{longest_string}}')
        return "\n".join(text)

    def filter(self, filter: Union['_', Callable[[_AbstractParam], bool]]):  # noqa F821 # type: ignore
        """Filter the parameters in the ParamsDict according to the provided filter."""
        params = ParamDict()

        if hasattr(filter, "apply"):
            params.update((name, v) for name, v in self.items() if filter.apply(v))
        else:
            params.update((name, v) for name, v in self.items() if filter(name, v))

        return params

    def with_prefix(self, pre):
        return ParamDict(
            {f"{pre}{k}": v for k, v in self.items()}
        )


# Register ParamsDict in the JAX pytree flatten/unflatten util ########################################################


def flatten_paramsdict(params: ParamDict) -> Tuple[Any, Any]:
    seen = {}

    for k, v in params.items():
        uid = id(v)
        if uid not in seen:
            seen[uid] = [
                v,
                (k,)
            ]
        else:
            seen[uid][1] += (k,)

    values = [p[0] for p in seen.values()]
    keys = tuple(p[1] for p in seen.values())

    return values, keys


def flatten_paramsdict_with_keys(params: ParamDict) -> Tuple[Any, Any]:
    values, keys = flatten_paramsdict(params)

    return tuple(zip(keys, values)), keys


def unflatten_paramsdict(aux_data: Any, flatten_data: Any) -> ParamDict:
    if len(flatten_data) == 0:
        return ParamDict()

    indices, keys = zip(*((i, key) for i, keys in enumerate(aux_data) for key in keys))
    values = (flatten_data[i] for i in indices)

    return ParamDict(zip(keys, values))


jax.tree_util.register_pytree_with_keys(
    ParamDict, flatten_paramsdict_with_keys, unflatten_paramsdict, flatten_paramsdict
)
