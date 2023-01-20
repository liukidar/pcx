from typing import Optional
import equinox as eqx
import jax.tree_util as jt

from ..core import (
    Module as _Module,
    BaseVar,
    TrainVar,
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
        rkey=DEFAULT_GENERATOR,
        **kwargs,
    ):
        super().__init__()
        self.nn = jt.tree_map(
            lambda w: LinkVar(w) if filter(w) else w,
            cls(*args, **kwargs, key=rkey()),
        )

    def __call__(self, *args, rkey=DEFAULT_GENERATOR, **kwargs):
        return self.nn(*args, **kwargs, key=rkey())

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
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(eqx.nn.Linear, in_features, out_features, bias)
