__all__ = ["NodeVar", "LinkVar", "ParameterCache", "Parameter"]

from typing import Optional, Callable
import jax

from ..core.parameters import Parameter, reduce_id, reduce_none, ParameterCache


class NodeVar(Parameter):
    def __init__(
        self,
        tensor: Optional[jax.Array] = None,
        reduce: Optional[Callable[[jax.Array], jax.Array]] = reduce_id,
        frozen: bool = False,
    ):
        super().__init__(tensor, reduce)
        self.frozen = frozen


class LinkVar(Parameter):
    def __init__(
        self,
        tensor: Optional[jax.Array] = None,
        reduce: Optional[Callable[[jax.Array], jax.Array]] = reduce_none,
    ):
        super().__init__(tensor, reduce)
