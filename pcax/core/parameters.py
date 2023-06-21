__all__ = [
    "_BaseParameter",
    "Parameter",
    "ParameterRef",
    "ParamsDict",
    "ParameterCache",
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


def get_jax_value(x: Union[jax.Array, "_BaseParameter"]):
    """Returns JAX value encapsulated in the input argument."""
    if isinstance(x, _BaseParameter):
        return x.value
    else:
        return x


def reduce_none(x: Optional[jax.Array]) -> jax.Array:
    return x[0] if x is not None else None


def reduce_mean(x: jax.Array) -> jax.Array:
    return x.mean(axis=0)


def reduce_id(x: jax.Array) -> jax.Array:
    return x


class _AbstractParameterMeta(abc.ABCMeta):
    """
    Metaclass to register all parameters in the JAX pytree flatten/unflatten util.
    """
    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        jax.tree_util.register_pytree_with_keys(
            cls,
            flatten_func=_AbstractParameterMeta.flatten_parameter,
            flatten_with_keys=_AbstractParameterMeta.flatten_parameter_with_keys,
            unflatten_func=_AbstractParameterMeta.unflatten_parameter
        )

        return cls

    @staticmethod
    def flatten_parameter(var: '_AbstractParameter') -> Tuple[Any, '_AbstractParameter']:
        return (var.value,), var

    @staticmethod
    def flatten_parameter_with_keys(var: '_AbstractParameter') -> Tuple[Any, '_AbstractParameter']:
        return (("value", var.value),), var

    @staticmethod
    def unflatten_parameter(var: '_AbstractParameter', data: Any) -> '_AbstractParameter':
        var.value = data[0]
        return var


class _AbstractParameter(metaclass=_AbstractParameterMeta):
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

    @abc.abstractmethod
    def reduce(self):
        """Method called by Vectorize to reduce a multiple-device (or batched in case of vectorization)
        value to a single device."""
        raise NotImplementedError("Pure method")


class _BaseParameter(_AbstractParameter):
    """The abstract base class to represent variables."""

    def __init__(
        self,
        value: Optional[jax.Array] = None,
        reduce: Optional[Callable[[jax.Array], jax.Array]] = reduce_none
    ):
        self._value = value
        self._reduce = reduce

    def __move__(self, __other: Optional['_BaseParameter'] = None) -> '_BaseParameter':
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
        """Method called by Vectorize to reduce a multiple-device (or batched in case of vectoriaation)
        value to a single device."""
        self.value = self._reduce(self.value)

    def __repr__(self):
        rvalue = type(self.value)  # re.sub("[\n]+", "\n", repr(self.value))
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
        return self.value.__eq__(get_jax_value(__other))  # noqa: E704

    def __ne__(self, __other):
        return self.value.__ne__(get_jax_value(__other))  # noqa: E704

    def __lt__(self, __other):
        return self.value.__lt__(get_jax_value(__other))  # noqa: E704

    def __le__(self, __other):
        return self.value.__le__(get_jax_value(__other))  # noqa: E704

    def __gt__(self, __other):
        return self.value.__gt__(get_jax_value(__other))  # noqa: E704

    def __ge__(self, __other):
        return self.value.__ge__(get_jax_value(__other))  # noqa: E704

    def __add__(self, __other):
        return self.value.__add__(get_jax_value(__other))  # noqa: E704

    def __radd__(self, __other):
        return self.value.__radd__(get_jax_value(__other))  # noqa: E704

    def __sub__(self, __other):
        return self.value.__sub__(get_jax_value(__other))  # noqa: E704

    def __rsub__(self, __other):
        return self.value.__rsub__(get_jax_value(__other))  # noqa: E704

    def __mul__(self, __other):
        return self.value.__mul__(get_jax_value(__other))  # noqa: E704

    def __rmul__(self, __other):
        return self.value.__rmul__(get_jax_value(__other))  # noqa: E704

    def __div__(self, __other):
        return self.value.__div__(get_jax_value(__other))  # noqa: E704

    def __rdiv__(self, __other):
        return self.value.__rdiv__(get_jax_value(__other))  # noqa: E704

    def __truediv__(self, __other):
        return self.value.__truediv__(get_jax_value(__other))  # noqa: E704

    def __rtruediv__(self, __other):
        return self.value.__rtruediv__(get_jax_value(__other))  # noqa: E704

    def __floordiv__(self, __other):
        return self.value.__floordiv__(get_jax_value(__other))  # noqa: E704

    def __rfloordiv__(self, __other):
        return self.value.__rfloordiv__(get_jax_value(__other))  # noqa: E704

    def __divmod__(self, __other):
        return self.value.__divmod__(get_jax_value(__other))  # noqa: E704

    def __rdivmod__(self, __other):
        return self.value.__rdivmod__(get_jax_value(__other))  # noqa: E704

    def __mod__(self, __other):
        return self.value.__mod__(get_jax_value(__other))  # noqa: E704

    def __rmod__(self, __other):
        return self.value.__rmod__(get_jax_value(__other))  # noqa: E704

    def __pow__(self, __other):
        return self.value.__pow__(get_jax_value(__other))  # noqa: E704

    def __rpow__(self, __other):
        return self.value.__rpow__(get_jax_value(__other))  # noqa: E704

    def __matmul__(self, __other):
        return self.value.__matmul__(get_jax_value(__other))  # noqa: E704

    def __rmatmul__(self, __other):
        return self.value.__rmatmul__(get_jax_value(__other))  # noqa: E704

    def __and__(self, __other):
        return self.value.__and__(get_jax_value(__other))  # noqa: E704

    def __rand__(self, __other):
        return self.value.__rand__(get_jax_value(__other))  # noqa: E704

    def __or__(self, __other):
        return self.value.__or__(get_jax_value(__other))  # noqa: E704

    def __ror__(self, __other):
        return self.value.__ror__(get_jax_value(__other))  # noqa: E704

    def __xor__(self, __other):
        return self.value.__xor__(get_jax_value(__other))  # noqa: E704

    def __rxor__(self, __other):
        return self.value.__rxor__(get_jax_value(__other))  # noqa: E704

    def __lshift__(self, __other):
        return self.value.__lshift__(get_jax_value(__other))  # noqa: E704

    def __rlshift__(self, __other):
        return self.value.__rlshift__(get_jax_value(__other))  # noqa: E704

    def __rshift__(self, __other):
        return self.value.__rshift__(get_jax_value(__other))  # noqa: E704

    def __rrshift__(self, __other):
        return self.value.__rrshift__(get_jax_value(__other))  # noqa: E704

    def __round__(self, ndigits=None):
        return self.value.__round__(ndigits)  # noqa: E704

    def __getitem__(self, __idx):
        return self.value.__getitem__(__idx)

    def __jax_array__(self):
        return self.value

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


class Parameter(_BaseParameter):
    """A trainable variable."""

    def __init__(
        self,
        value: Optional[jax.Array] = None,
        reduce: Optional[Callable[[jax.Array], jax.Array]] = reduce_none,
    ):
        """Parameter constructor.

        Args:
            value: the initial value of the Parameter.
            reduce: a function that takes an array of shape ``(n, *dims)`` and returns one of shape ``(*dims)``. Used to
                    combine the multiple states produced in an pcax.Vectorize or an pcax.Parallel call.
        """
        super().__init__(value, reduce)
        self._value = value


class ParameterRef(_AbstractParameter):
    """A state variable that references a trainable variable for assignment.

    ParameterRef are used by optimizers to keep references to trainable variables. This is necessary to differentiate
    them from the optimizer own training variables if any."""

    def __init__(self, ref: Parameter):
        """ParameterRef constructor.

        Args:
            ref: the Parameter to keep the reference of.
        """
        super().__init__()
        self.ref = ref

    def __move__(self, target=None) -> 'ParameterRef':
        if target is None:
            new_var = copy.copy(self)
        else:
            new_var = target
            new_var.ref = self.ref
            new_var.value = self.value
        self.ref = None
        self.value = None

        return new_var

    @property
    def value(self) -> jax.Array:
        """The value stored in the referenced Parameter, it can be read or written."""
        return None

    @value.setter
    def value(self, value: jax.Array):
        return

    def reduce(self):
        return

    def __repr__(self):
        return f"{self.__class__.__name__}(ref={repr(self.ref)})"


class ParameterCache(_BaseParameter, ParameterRef):
    def __init__(
        self, param: Parameter
    ):
        _BaseParameter.__init__(self, {}, None)
        ParameterRef.__init__(self, param)

    def __getitem__(self, key: str):
        return self._value[key]

    def __setitem__(self, key: str, value: jax.Array):
        self._value[key] = value

    def __contains__(self, key: str):
        return key in self._value

    def clear(self):
        self._value = {}

    def reduce(self):
        for k, v in self.value.items():
            self.value[k] = self.ref._reduce(v)

    def __repr__(self):
        rvalue = re.sub("[\n]+", "\n", repr(self.value))
        t = f"{self.__class__.__name__}({rvalue})"
        return f"{t[:-1]}"


class ParamsDict(Dict[str, _AbstractParameter]):
    """A ParamsDict is a dictionary (name, var) with some additional methods to make manipulation of collections of
    variables easy. A ParamsDict is ordered by insertion order. It is the object returned by Module.vars() and used
    as input by many modules: optimizers, Jit, etc..."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __move__(self, __target: Optional['ParamsDict' | Dict[str, _AbstractParameter]] = None):
        params = __target or ParamsDict()

        seen = {}
        for k, v in self.items():
            uid = id(v)
            if uid not in seen:
                seen[uid] = move(v, params[k] if __target else None)

            params[k] = seen[uid]

        return params

    def __add__(self, __other: "ParamsDict") -> "ParamsDict":
        """Overloaded add operator to merge two ParamsDicts together."""
        params = ParamsDict(self)
        params.update(__other)
        return params

    def __sub__(self, __other: "ParamsDict") -> "ParamsDict":
        """Overloaded add operator to merge two ParamsDicts together."""
        params = ParamsDict()
        other_ids = set(map(lambda x: id(x), __other.values()))

        for k, v in self.items():
            if id(v) not in other_ids:
                params[k] = v

        return params

    def __iter__(self) -> Iterator[_AbstractParameter]:
        """Create an iterator that iterates over the variables (dict values) and visit them only once.
        If a variable has two names, for example in the case of weight sharing, this iterator yields the variable only
        once."""
        seen = set()
        for v in self.values():
            uid = id(v)
            if uid not in seen:
                seen.add(uid)
                yield v

    def __setitem__(self, __key: str, __value: _AbstractParameter):
        """Overload bracket assignment to catch potential conflicts during assignment."""
        if __key in self and self[__key] is not __value:
            raise ValueError("Name conflicts when appending to ParamsDict", __key)
        dict.__setitem__(self, __key, __value)

    def update(self, other: Union["ParamsDict", Iterable[Tuple[str, _AbstractParameter]]]):
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

    def filter(self, filter: Union['_', Callable[[_AbstractParameter], bool]]):  # noqa F821 # type: ignore
        params = ParamsDict()

        if hasattr(filter, "apply"):
            params.update((name, v) for name, v in self.items() if filter.apply(v))
        else:
            params.update((name, v) for name, v in self.items() if filter(name, v))

        return params

    def rename(self, name):
        return ParamsDict(
            {re.sub(r"\(.*?\)", name, k, count=1): v for k, v in self.items()}
        )


def flatten_paramsdict(params: ParamsDict) -> Tuple[Any, Any]:
    seen = {}

    for k, v in params.items():
        uid = id(v)
        if uid not in seen:
            seen[uid] = (
                v,
                (k,)
            )
        else:
            seen[uid][1] += (k,)

    values = [p[0] for p in seen.values()]
    keys = tuple(p[1] for p in seen.values())

    return values, keys


def flatten_paramsdict_with_keys(params: ParamsDict) -> Tuple[Any, Any]:
    values, keys = flatten_paramsdict(params)

    return tuple(zip(keys, values)), keys


def unflatten_paramsdict(aux_data: Any, flatten_data: Any) -> ParamsDict:
    if len(flatten_data) == 0:
        return ParamsDict()

    indices, keys = zip(*((i, key) for i, keys in enumerate(aux_data) for key in keys))
    values = (flatten_data[i] for i in indices)

    return ParamsDict(zip(keys, values))


jax.tree_util.register_pytree_with_keys(
    ParamsDict, flatten_paramsdict_with_keys, unflatten_paramsdict, flatten_paramsdict
)
