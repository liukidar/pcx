__all__ = ["NodeParam", "LayerParam", "ParamCache", "Param"]

from typing import Optional, Callable
import jax

from ..core.parameters import Param, reduce_id, reduce_none, ParamCache


class NodeParam(Param):
    def __init__(
        self,
        tensor: Optional[jax.Array] = None,
        reduce: Optional[Callable[[jax.Array], jax.Array]] = reduce_id,
        frozen: bool = False,
    ):
        super().__init__(tensor, reduce)
        self.frozen = frozen


class LayerParam(Param):
    def __init__(
        self,
        tensor: Optional[jax.Array] = None,
        reduce: Optional[Callable[[jax.Array], jax.Array]] = reduce_none,
        frozen: bool = False,
    ):
        super().__init__(tensor, reduce)
        self.frozen = frozen
