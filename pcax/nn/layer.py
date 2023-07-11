from typing import Optional, Tuple, Union, Sequence
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
        self.in_features = in_features
        self.out_features = out_features
        super().__init__(eqx.nn.Linear, in_features, out_features, bias, key=RKG())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


class LayerNorm(Layer):
    def __init__(
        self,
        shape: Optional[Tuple[int, ...]] = None,
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
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True
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
            key=RKG()
        )


class Conv2d(Conv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True
    ):
        super().__init__(2, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, use_bias)


class MaxPool2d(Layer):
    def __init__(
        self,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        use_ceil: bool = False,
        **kwargs
    ):
        super().__init__(eqx.nn.MaxPool2d, kernel_size, stride, padding, use_ceil, **kwargs)


class AvgPool2d(Layer):
    def __init__(
        self,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        use_ceil: bool = False,
        **kwargs
    ):
        super().__init__(eqx.nn.AvgPool2d, kernel_size, stride, padding, use_ceil, **kwargs)
