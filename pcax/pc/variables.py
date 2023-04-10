__all__ = ["CachedVar", "NodeVar", "TrainVar", "LinkVar", "StateVar"]

from typing import Optional, Callable, Tuple, Union
import jax

from ..core.structure import BaseVar, TrainVar, reduce_id, reduce_none, StateVar


class CachedVar(BaseVar):
    def __init__(self):
        super().__init__(NotImplemented)
        self._cache = {}

    @property
    def value(self) -> None:
        raise ValueError(
            "CachedVar is a collection of variables and does not have a specific value, use [] operator to access"
            "a specific variable."
        )

    @value.setter
    def value(self, tensor: jax.Array):
        raise ValueError(
            "Direct assignment not allowed, use __setitem__ to update a CachedVar."
        )

    def dump(self):
        """Splits the variable into the set of dynamic parameters (differentiable and not) and static parameters.
        The parameters not dumped are considered constant throught the existance of the variable."""

        return (
            None,
            tuple(self._cache.values()),
            tuple(self._cache.keys()),
        )

    def load(
        self,
        differentiable: Optional[jax.Array] = None,
        dynamic: Optional[Tuple[Union[jax.Array, Tuple[jax.Array]]]] = None,
        static: Tuple[Tuple[str]] = None,
        reduce: bool = False,
    ):
        static = static or self._cache.keys()
        dynamic = dynamic or (None,) * len(static)
        self._cache = dict(zip(static, dynamic))

    def clear(self):
        self._cache = {}

    def __getitem__(self, key):
        return self._cache[key]

    def __setitem__(self, key, value):
        self._cache[key] = value

    def __contains__(self, key):
        return key in self._cache


class NodeVar(TrainVar):
    def __init__(
        self,
        tensor: Optional[jax.Array] = None,
        reduce: Optional[Callable[[jax.Array], jax.Array]] = reduce_id,
        frozen: bool = False,
    ):
        super().__init__(tensor, reduce)
        self.frozen = frozen

    def dump(self):
        """Splits the variable into the set of dynamic parameters (differentiable and not) and static parameters.
        The parameters not dumped are considered constant throught the existance of the variable."""

        return self.value, None, self.frozen

    def load(self, value, dynamic=None, static=None, reduce=False):
        if reduce:
            self.reduce(value)
        else:
            self.value = value

        if static is not None:
            self.frozen = static


class LinkVar(TrainVar):
    def __init__(
        self,
        tensor: Optional[jax.Array] = None,
        reduce: Optional[Callable[[jax.Array], jax.Array]] = reduce_none,
    ):
        super().__init__(tensor, reduce)

    def assign(self, other):
        """Assign a new value to the variable. Kept for compatibility with the objax.io package."""
        if not self.shape == other.shape:
            raise ValueError(
                f"Shape mismatch: Cannot assign {other.shape} to {self.value.shape} for LinkVar {self.name}."
            )
        self.value = other
