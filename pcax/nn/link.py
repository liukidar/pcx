from typing import Optional, Tuple
import equinox as eqx
import jax.tree_util as jt


from ..core import (
    Module as _Module,
    BaseVar,
    VarCollection,
    DEFAULT_GENERATOR,
)
from ..core.filter import _

from ..pc import LinkVar


class Link(_Module):
    def __init__(
        self,
        cls,
        *args,
        filter=eqx.filters.is_array,
        **kwargs,
    ):
        super().__init__()
        self.nn = jt.tree_map(
            lambda w: LinkVar(w) if filter(w) else w,
            cls(*args, **kwargs),
        )

    def __call__(self, *args, key=None, **kwargs):
        return self.nn(*args, **kwargs, key=key)

    def vars(self, filter: Optional[_] = None, scope: str = "") -> VarCollection:
        vc = super().vars(scope=scope)
        scope += f"({self.__class__.__name__}).nn."
        for k, v in self.nn.__dict__.items():
            if isinstance(v, BaseVar):
                vc[scope + k] = v

        if filter is not None:
            vc = vc.filter(filter)

        return vc


class Linear(Link):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(eqx.nn.Linear, in_features, out_features, bias, key=DEFAULT_GENERATOR())


class LayerNorm(Link):
    def __init__(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
    ):
        super().__init__(eqx.nn.LayerNorm, shape, eps, elementwise_affine)
