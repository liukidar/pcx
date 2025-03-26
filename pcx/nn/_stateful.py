__all__ = ["StateParam", "StatefulLayer", "BatchNorm"]

from typing import Hashable, Sequence

import jax.tree_util as jtu
import equinox as eqx

from ..core._module import Module
from ..core._parameter import BaseParam, Param
from ..core._static import StaticParam
from ._parameter import LayerParam


class StateParam(Param):
    pass


class StatefulLayer(Module):
    def __init__(self, cls, *args, filter=eqx._filters.is_array, **kwargs) -> None:
        super().__init__()

        self.nn, self.state = eqx.nn.make_with_state(cls)(*args, **kwargs)

        self.nn = jtu.tree_map(
            lambda w: LayerParam(w) if filter(w) else StaticParam(w),
            self.nn,
        )

        # print(self.nn)
        # print(self.state)

        # We opt for a single StateParam to encapsule all the state data. This limits the
        # masking that can be done on it (as you either select all or nothing). For a for
        # fine-grained selection, a per-array StateParam approach must be used.
        #
        # self.state = jtu.tree_map(
        #     lambda w: StateParam(w) if filter(w) else StaticParam(w),
        #     self.state,
        # )
        #
        self.state = StateParam(self.state)

    def __call__(self, *args, key=None, **kwargs):
        _nn = jtu.tree_map(
            lambda w: w.get() if isinstance(w, BaseParam) else w,
            self.nn,
            is_leaf=lambda w: isinstance(w, BaseParam),
        )

        _r, _state = _nn(*args, self.state.get(), **kwargs, key=key)

        # Alternative per-array StateParam approach
        #
        # jtu.tree_map(
        #     lambda p, v: p.set(v) if isinstance(p, StateParam) else None,
        #     self.state,
        #     _state,
        #     is_leaf=lambda w: isinstance(w, BaseParam),
        # )
        #
        self.state.set(_state)

        return _r


class BatchNorm(StatefulLayer):
    def __init__(
        self,
        input_size: int,
        axis_name: Hashable | Sequence[Hashable],
        eps: float = 1e-05,
        channelwise_affine: bool = True,
        momentum: float = 0.1,
        inference: bool = False,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            eqx.nn.BatchNorm,
            input_size,
            axis_name,
            eps,
            channelwise_affine,
            momentum,
            inference,
            dtype,
            **kwargs,
        )
