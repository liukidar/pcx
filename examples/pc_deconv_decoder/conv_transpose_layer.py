from typing import Sequence, Union

import equinox as eqx

from pcax.nn import Layer
from pcax.core import RandomKeyGenerator, RKG


class ConvTranspose(Layer):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]] = 0,
        output_padding: Union[int, Sequence[int]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        rkg: RandomKeyGenerator = RKG,
    ):
        super().__init__(
            eqx.nn.ConvTranspose,
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            padding_mode=padding_mode,
            key=rkg(),
        )
