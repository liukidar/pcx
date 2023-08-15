from typing import Optional, Tuple
import equinox as eqx
import jax.tree_util as jt


from ..core import (
    Module,
    RKG,
)

from ..pc import LayerParam


class Layer(Module):
    def __init__(
        self,
        cls,
        *args,
        filter=eqx._filters.is_array,
        **kwargs,
    ):
        super().__init__()
        self.nn = jt.tree_map(
            lambda w: LayerParam(w) if filter(w) else w,
            cls(*args, **kwargs),
        )

    def __call__(self, *args, key=None, **kwargs):
        return self.nn(*args, **kwargs, key=key)


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(
            eqx.nn.Linear, in_features, out_features, bias, key=RKG()
        )


class LayerNorm(Layer):
    def __init__(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
    ):
        super().__init__(eqx.nn.LayerNorm, shape, eps, elementwise_affine)
