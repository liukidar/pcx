__all__ = [
    # Variable
    "BaseVar",
    "TrainVar",
    "TrainRef",
    "StateVar",
    "RandomState",
    "VarCollection",
    # Module
    "Module",
    "ModuleList",
    "Function",
]

import abc
import re
from typing import List, Union, Tuple, Optional, Iterable, Dict, Iterator, Callable, Any

import jax
import jax.random as jr
import numpy as np

from .util import repr_function
from .filter import _

########################################################################################################################
#
# VARIABLES
#
########################################################################################################################


def get_jax_value(x: Union[jax.Array, "BaseVar"]):
    """Returns JAX value encapsulated in the input argument."""
    if isinstance(x, BaseVar):
        return x.value
    else:
        return x


def reduce_none(x: Optional[jax.Array]) -> jax.Array:
    return x[0] if x is not None else None


def reduce_mean(x: jax.Array) -> jax.Array:
    return x.mean(axis=0)


def reduce_id(x: jax.Array) -> jax.Array:
    return x


class BaseVar(abc.ABC):
    """The abstract base class to represent variables."""

    def __init__(
        self, reduce: Optional[Callable[[jax.Array], jax.Array]] = reduce_none
    ):
        """Constructor for BaseVar class.

        Args:
            reduce: a function that takes an array of shape ``(n, *dims)`` and returns one of shape ``(*dims)``. Used to
                    combine the multiple states produced in an pcax.Vectorize or an pcax.Parallel call.
        """
        self._reduce = reduce

    @property
    @abc.abstractmethod
    def value(self) -> jax.Array:
        raise NotImplementedError("Pure method")

    @value.setter
    @abc.abstractmethod
    def value(self, tensor: jax.Array):
        raise NotImplementedError("Pure method")

    def reduce(self, tensors: jax.Array):
        """Method called by Parallel and Vectorize to reduce a multiple-device (or batched in case of vectoriaation)
        value to a single device."""
        self.value = self._reduce(tensors)

    def dump(self):
        """Splits the variable into the set of dynamic parameters (differentiable and not) and static parameters.
        The parameters not dumped are considered constant throught the existance of the variable."""

        return self.value, None, None

    def load(self, value, dynamic=None, static=None, reduce=False):
        if reduce:
            self.reduce(value)
        else:
            self.value = value

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

    def __eq__(self, other):
        return self.value.__eq__(get_jax_value(other))  # noqa: E704

    def __ne__(self, other):
        return self.value.__ne__(get_jax_value(other))  # noqa: E704

    def __lt__(self, other):
        return self.value.__lt__(get_jax_value(other))  # noqa: E704

    def __le__(self, other):
        return self.value.__le__(get_jax_value(other))  # noqa: E704

    def __gt__(self, other):
        return self.value.__gt__(get_jax_value(other))  # noqa: E704

    def __ge__(self, other):
        return self.value.__ge__(get_jax_value(other))  # noqa: E704

    def __add__(self, other):
        return self.value.__add__(get_jax_value(other))  # noqa: E704

    def __radd__(self, other):
        return self.value.__radd__(get_jax_value(other))  # noqa: E704

    def __sub__(self, other):
        return self.value.__sub__(get_jax_value(other))  # noqa: E704

    def __rsub__(self, other):
        return self.value.__rsub__(get_jax_value(other))  # noqa: E704

    def __mul__(self, other):
        return self.value.__mul__(get_jax_value(other))  # noqa: E704

    def __rmul__(self, other):
        return self.value.__rmul__(get_jax_value(other))  # noqa: E704

    def __div__(self, other):
        return self.value.__div__(get_jax_value(other))  # noqa: E704

    def __rdiv__(self, other):
        return self.value.__rdiv__(get_jax_value(other))  # noqa: E704

    def __truediv__(self, other):
        return self.value.__truediv__(get_jax_value(other))  # noqa: E704

    def __rtruediv__(self, other):
        return self.value.__rtruediv__(get_jax_value(other))  # noqa: E704

    def __floordiv__(self, other):
        return self.value.__floordiv__(get_jax_value(other))  # noqa: E704

    def __rfloordiv__(self, other):
        return self.value.__rfloordiv__(get_jax_value(other))  # noqa: E704

    def __divmod__(self, other):
        return self.value.__divmod__(get_jax_value(other))  # noqa: E704

    def __rdivmod__(self, other):
        return self.value.__rdivmod__(get_jax_value(other))  # noqa: E704

    def __mod__(self, other):
        return self.value.__mod__(get_jax_value(other))  # noqa: E704

    def __rmod__(self, other):
        return self.value.__rmod__(get_jax_value(other))  # noqa: E704

    def __pow__(self, other):
        return self.value.__pow__(get_jax_value(other))  # noqa: E704

    def __rpow__(self, other):
        return self.value.__rpow__(get_jax_value(other))  # noqa: E704

    def __matmul__(self, other):
        return self.value.__matmul__(get_jax_value(other))  # noqa: E704

    def __rmatmul__(self, other):
        return self.value.__rmatmul__(get_jax_value(other))  # noqa: E704

    def __and__(self, other):
        return self.value.__and__(get_jax_value(other))  # noqa: E704

    def __rand__(self, other):
        return self.value.__rand__(get_jax_value(other))  # noqa: E704

    def __or__(self, other):
        return self.value.__or__(get_jax_value(other))  # noqa: E704

    def __ror__(self, other):
        return self.value.__ror__(get_jax_value(other))  # noqa: E704

    def __xor__(self, other):
        return self.value.__xor__(get_jax_value(other))  # noqa: E704

    def __rxor__(self, other):
        return self.value.__rxor__(get_jax_value(other))  # noqa: E704

    def __lshift__(self, other):
        return self.value.__lshift__(get_jax_value(other))  # noqa: E704

    def __rlshift__(self, other):
        return self.value.__rlshift__(get_jax_value(other))  # noqa: E704

    def __rshift__(self, other):
        return self.value.__rshift__(get_jax_value(other))  # noqa: E704

    def __rrshift__(self, other):
        return self.value.__rrshift__(get_jax_value(other))  # noqa: E704

    def __round__(self, ndigits=None):
        return self.value.__round__(ndigits)  # noqa: E704

    def __getitem__(self, idx):
        return self.value.__getitem__(idx)

    def __jax_array__(self):
        return self.value

    def __array__(self, dtype=None):
        return self.value.__array__(dtype)

    def __bool__(self):
        raise TypeError(
            "To prevent accidental errors variables can not be used as Python bool. "
            "To check if variable is `None` use `is None` or `is not None` instead."
        )

    def __getattr__(self, name):
        return getattr(self.value, name)

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


class TrainVar(BaseVar):
    """A trainable variable."""

    def __init__(
        self,
        value: Optional[jax.Array] = None,
        reduce: Optional[Callable[[jax.Array], jax.Array]] = reduce_none,
    ):
        """TrainVar constructor.

        Args:
            value: the initial value of the TrainVar.
            reduce: a function that takes an array of shape ``(n, *dims)`` and returns one of shape ``(*dims)``. Used to
                    combine the multiple states produced in an pcax.Vectorize or an pcax.Parallel call.
        """
        self._value = value
        super().__init__(reduce)

    @property
    def value(self) -> jax.Array:
        """The value is read only as a safety measure to avoid accidentally making TrainVar non-differentiable.
        You can write a value to a TrainVar by using assign."""
        return self._value

    @value.setter
    def value(self, value: jax.Array):
        self._value = value


class TrainRef(BaseVar):
    """A state variable that references a trainable variable for assignment.

    TrainRef are used by optimizers to keep references to trainable variables. This is necessary to differentiate
    them from the optimizer own training variables if any."""

    def __init__(self, ref: TrainVar):
        """TrainRef constructor.

        Args:
            ref: the TrainVar to keep the reference of.
        """
        self.ref = ref
        super().__init__(None)

    @property
    def value(self) -> jax.Array:
        """The value stored in the referenced TrainVar, it can be read or written."""
        return self.ref.value

    @value.setter
    def value(self, value: jax.Array):
        self.ref.value = value

    def __repr__(self):
        return f"{self.__class__.__name__}(ref={repr(self.ref)})"

    def dump(self):
        return (None, None, None)

    def load(self, value, dynamic=None, static=None, reduce=False):
        return


class StateVar(BaseVar):
    """StateVar are variables that can be updated manually.
    For example, the mean and variance statistics in BatchNorm are StateVar."""

    def __init__(
        self,
        value: jax.Array,
        reduce: Optional[Callable[[jax.Array], jax.Array]] = reduce_none,
    ):
        """StateVar constructor.

        Args:
            tensor: the initial value of the StateVar.
            reduce: a function that takes an array of shape ``(n, *dims)`` and returns one of shape ``(*dims)``.
                    Used to combine the multiple states produced in an pcax.Vectorize or an pcax.Parallel call.
        """
        self._value = value
        super().__init__(reduce)

    @property
    def value(self) -> jax.Array:
        """The value stored in the StateVar, it can be read or written."""
        return self._value

    @value.setter
    def value(self, value: jax.Array):
        self._value = value


class RandomState(StateVar):
    """RandomState are variables that track the random generator state. They are meant to be used internally.
    Currently only the random.Generator module uses them."""

    def __init__(self, seed: int):
        """RandomState constructor.

        Args:
            seed: the initial seed of the random number generator.
        """
        super().__init__(jr.PRNGKey(seed), reduce_none)

    def seed(self, seed: int):
        """Sets a new random seed.

        Args:
            seed: the new initial seed of the random number generator.
        """
        self.value = jr.PRNGKey(seed)

    def split(self, n: int) -> List[jax.Array]:
        """Create multiple seeds from the current seed. This is used internally by Parallel and Vectorize to ensure
        that random numbers are different in parallel threads.

        Args:
            n: the number of seeds to generate.
        """
        values = jr.split(self.value, n + 1)
        self._value = values[0]
        return values[1:]

    def dump(self):
        """Splits the variable into the set of dynamic parameters (differentiable and not) and static parameters.
        The parameters not dumped are considered constant throught the existance of the variable."""

        return None, self.value, None

    def load(self, value, dynamic=None, static=None, reduce=False):
        if dynamic is not None:
            if reduce:
                self.reduce(dynamic)
            else:
                self.value = dynamic


class VarCollection(Dict[str, BaseVar]):
    """A VarCollection is a dictionary (name, var) with some additional methods to make manipulation of collections of
    variables easy. A VarCollection is ordered by insertion order. It is the object returned by Module.vars() and used
    as input by many modules: optimizers, Jit, etc..."""

    def __add__(self, other: "VarCollection") -> "VarCollection":
        """Overloaded add operator to merge two VarCollections together."""
        vc = VarCollection(self)
        vc.update(other)
        return vc

    def __sub__(self, other: "VarCollection") -> "VarCollection":
        """Overloaded add operator to merge two VarCollections together."""
        vc = VarCollection()
        other_ids = set(map(lambda x: id(x), other.values()))

        for k, v in self.items():
            if id(v) not in other_ids:
                vc[k] = v

        return vc

    def __iter__(self) -> Iterator[BaseVar]:
        """Create an iterator that iterates over the variables (dict values) and visit them only once.
        If a variable has two names, for example in the case of weight sharing, this iterator yields the variable only
        once."""
        seen = set()
        for v in self.values():
            if id(v) not in seen:
                seen.add(id(v))
                yield v

    def __setitem__(self, key: str, value: BaseVar):
        """Overload bracket assignment to catch potential conflicts during assignment."""
        if key in self and self[key] != value:
            raise ValueError("Name conflicts when appending to VarCollection", key)
        dict.__setitem__(self, key, value)

    def update(self, other: Union["VarCollection", Iterable[Tuple[str, BaseVar]]]):
        """Overload dict.update method to catch potential conflicts during assignment."""
        if not isinstance(other, self.__class__):
            other = list(other)
        else:
            other = other.items()
        conflicts = set()
        for k, v in other:
            if k in self:
                if self[k] is not v:
                    conflicts.add(k)
            else:
                self[k] = v
        if conflicts:
            raise ValueError(
                f"Name conflicts when combining VarCollection {sorted(conflicts)}"
            )

    def __str__(self, max_width=100):
        """Pretty print the contents of the VarCollection."""
        text = []
        total = count = 0
        longest_string = max((len(x) for x in self.keys()), default=20)
        longest_string = min(max_width, max(longest_string, 20))
        for name, v in self.items():
            size = np.prod(v.value.shape) if v.value.ndim else 1
            total += size
            count += 1
            text.append(f"{name:{longest_string}} {size:8d} {v.value.shape}")
        text.append(f'{f"+Total({count})":{longest_string}} {total:8d}')
        return "\n".join(text)

    def filter(self, filter: Union[_, Callable[[BaseVar], bool]]):
        vc = VarCollection()

        if isinstance(filter, _):
            vc.update((name, v) for name, v in self.items() if filter.apply(v))
        else:
            vc.update((name, v) for name, v in self.items() if filter(v))

        return vc

    def rename(self, name):
        return VarCollection(
            {re.sub(r"\(.*\)", name, k, count=1): v for k, v in self.items()}
        )

    def dump(self) -> Tuple[Tuple[jax.Array], Tuple[Any], Tuple[Any]]:
        r = tuple(zip(*(v.dump() for v in self)))
        if len(r) == 0:
            r = ((), (), ())

        return r

    def load(
        self,
        value: jax.Array,
        dynamic: Optional[Tuple[Any]] = None,
        static: Optional[Tuple[Any]] = None,
        reduce: bool = False,
    ) -> None:
        dynamic = dynamic or (None,) * len(value)
        static = static or (None,) * len(value)
        for v, di, dy, st in zip(self, value, dynamic, static):
            v.load(di, dy, st, reduce)


########################################################################################################################
#
# MODULES
#
########################################################################################################################


class Module:
    """A module is a container to associate variables and functions."""

    def vars(self, filter: Optional[_] = None, scope: str = "") -> VarCollection:
        """Collect all the variables (and their names) contained in the module and its submodules.
        Important: Variables and modules stored Python structures such as dict or list are not collected. See ModuleList
        if you need such a feature.

        Args:
            scope: string to prefix to the variable names.
        Returns:
            A VarCollection of all the variables.
        """
        vc = VarCollection()
        scope += f"({self.__class__.__name__})."
        for k, v in self.__dict__.items():
            if isinstance(v, BaseVar):
                vc[scope + k] = v
            elif isinstance(v, Module):
                if k == "__wrapped__":
                    vc.update(v.vars(scope=scope[:-1]))
                else:
                    vc.update(v.vars(scope=scope + k))

        if filter is not None:
            vc = vc.filter(filter)

        return vc

    def __call__(self, *args, **kwargs):
        """Optional module __call__ method, typically a forward pass computation for standard primitives."""
        raise NotImplementedError


class ModuleList(Module, list):
    """This is a replacement for Python's list that provides a vars() method to return all the variables that it
    contains, including the ones contained in the modules and sub-modules in it."""

    def vars(self, scope: str = "") -> VarCollection:
        """Collect all the variables (and their names) contained in the list and its submodules.

        Args:
            scope: string to prefix to the variable names.
        Returns:
            A VarCollection of all the variables.
        """
        vc = VarCollection()
        scope += f"({self.__class__.__name__})"
        for p, v in enumerate(self):
            if isinstance(v, BaseVar):
                vc[f"{scope}[{p}]"] = v
            elif isinstance(v, Module):
                vc.update(v.vars(scope=f"{scope}[{p}]"))
        return vc

    def __getitem__(self, key: Union[int, slice]):
        value = list.__getitem__(self, key)
        if isinstance(key, slice):
            return ModuleList(value)
        return value

    def __repr__(self):
        def f(x):
            if not isinstance(x, Module) and callable(x):
                return repr_function(x)
            x = repr(x).split("\n")
            x = [x[0]] + ["  " + y for y in x[1:]]
            return "\n".join(x)

        entries = "\n".join(f"  [{i}] {f(x)}" for i, x in enumerate(self))
        return f"{self.__class__.__name__}(\n{entries}\n)"


class Function(Module):
    """Turn a function into a Module by keeping the vars it uses."""

    def __init__(self, f: Callable, vc: Optional[VarCollection] = None):
        """Function constructor.

        Args:
            f: the function or the module to represent.
            vc: the VarCollection of variables used by the function.
        """
        if vc is None:
            raise NotImplementedError

        if hasattr(f, "__name__"):
            self.vc = VarCollection((f"{{{f.__name__}}}{k}", v) for k, v in vc.items())
        else:
            self.vc = VarCollection(vc)
        self.__wrapped__ = f

    def __call__(self, *args, **kwargs):
        """Call the the function."""
        return self.__wrapped__(*args, **kwargs)

    def vars(self, scope: str = "") -> VarCollection:
        """Return the VarCollection of the variables used by the function."""
        if scope:
            return VarCollection((scope + k, v) for k, v in self.vc.items())
        return VarCollection(self.vc)

    @staticmethod
    def with_vars(vc: VarCollection):
        """Decorator which turns a function into a module using provided variable collection.

        Args:
            vc: the VarCollection of variables used by the function.
        """

        def from_function(f: Callable):
            return Function(f, vc)

        return from_function

    @staticmethod
    def auto_vars(f: Callable):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(f={repr_function(self.__wrapped__)})"
