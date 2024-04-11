__all__ = [
    "Layer",
    "Linear",
    "LayerNorm",
    "Conv",
    "Conv2d",
    "MaxPool2d",
    "AvgPool2d",
]


from typing import Tuple, Sequence

import jax.tree_util as jtu
import equinox as eqx

from ..core._module import Module
from ..core._random import RandomKeyGenerator, RKG
from ..core._parameter import BaseParam
from ..core._static import StaticParam
from ._parameter import LayerParam


########################################################################################################################
#
# LAYER
#
# pcax layers are a thin wrapper around equinox layers that replaces all jax.Arrays with LayerParam instances.
# In this file only stateless layers are implemented as they don't need any particular ad-hoc adaptation.
########################################################################################################################


# Core #################################################################################################################


class Layer(Module):
    def __init__(
        self,
        cls,
        *args,
        filter=eqx._filters.is_array,
        **kwargs,
    ):
        super().__init__()
        self.nn = jtu.tree_map(
            lambda w: LayerParam(w) if filter(w) else StaticParam(w),
            cls(*args, **kwargs),
        )

    def __call__(self, *args, key=None, **kwargs):
        # Can do this, since nn is stateless
        _nn = jtu.tree_map(
            lambda w: w.get() if isinstance(w, BaseParam) else w,
            self.nn,
            is_leaf=lambda w: isinstance(w, BaseParam),
        )

        return _nn(*args, **kwargs, key=key)


# Common Layers ########################################################################################################


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, rkg: RandomKeyGenerator = RKG):
        super().__init__(eqx.nn.Linear, in_features, out_features, bias, key=rkg())


class LayerNorm(Layer):
    def __init__(
        self,
        shape: Tuple[int, ...] | None = None,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
    ):
        super().__init__(eqx.nn.LayerNorm, shape, eps, elementwise_affine)


class Conv(Layer):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | Sequence[Tuple[int, int]] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        rkg: RandomKeyGenerator = RKG,
    ):
        super().__init__(
            eqx.nn.Conv,
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            key=rkg(),
        )


class Conv2d(Conv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | Sequence[Tuple[int, int]] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        rkg: RandomKeyGenerator = RKG,
    ):
        super().__init__(2, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, use_bias, rkg)


# Pooling ##############################################################################################################


class MaxPool2d(Layer):
    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | Sequence[Tuple[int, int]] = 0,
        use_ceil: bool = False,
        **kwargs,
    ):
        super().__init__(eqx.nn.MaxPool2d, kernel_size, stride, padding, use_ceil, **kwargs)


class AvgPool2d(Layer):
    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | Sequence[Tuple[int, int]] = 0,
        use_ceil: bool = False,
        **kwargs,
    ):
        super().__init__(eqx.nn.AvgPool2d, kernel_size, stride, padding, use_ceil, **kwargs)
