__all__ = [
    "ErrorVode",
    "apply_h_updates",
]

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.tree_util as jtu
from jaxtyping import PyTree

from ..core._parameter import BaseParam, DynamicParam
from ..core._random import RKG, RandomKeyGenerator
from ..core._static import static
from ..core._tree import tree_apply
from ._energy import se_energy
from ._energy_module import EnergyModule
from ._parameter import VodeParam
from ._vode import STATUS, Ruleset, Vode

########################################################################################################################
#
# ERROR VODE
#
# A Vode parameterised by its prediction error e = h - u instead of its value h. The value is reconstructed as
# h = sg(u) + e, so the energy, its minimisers and the local PC gradients are the same as for a Vode; what changes is
# what is stored. On a trained network |e| << |h|, and a Vode that stores h loses e to rounding when it forms h - u
# (absolute error ~ eps * |h|). An ErrorVode keeps e at relative precision eps.
#
########################################################################################################################

# Core #################################################################################################################


def _zeros_like(vode: Vode, key: str, value: jax.Array | None, rkg: RandomKeyGenerator) -> jax.Array | None:
    return None if value is None else jax.numpy.zeros_like(value)


class ErrorVode(Vode):
    """
    A Vode that stores the prediction error `e` instead of the value `h`.

    The value seen by the next layer is `h = stop_gradient(u) + e`, and the residual used by the energy is
    `e + (stop_gradient(u) - u)`, which equals `e` exactly while still depending on `u`. Differentiating the energy
    then gives exactly the local PC gradients, read from the stored errors without any cancellation:

    - w.r.t. `e_l`: `e_l - J_{l+1}^T e_{l+1}`, i.e. the gradient a Vode receives w.r.t. `h_l`;
    - w.r.t. the weights of layer `l`: the usual fixed-state PC weight gradient.

    The stop-gradient is what keeps the gradient local: without it, the gradient w.r.t. `e_l` would also flow through
    every layer above (Goemaere et al., 2025).

    Forward initialisation (`STATUS.INIT`) sets `e = 0`, i.e. `h = u`.

    NOTE: the gradient is the same as a Vode's, but applying it to `e` is not the same update. Moving `e_l` by `d_l`
    moves `h_l` by `d_l` plus the change in `u_l` caused by the layers below, so `optim.step` on ErrorVodes runs a
    different (still local) inference dynamics. Use `apply_h_updates` to apply an update as a step in `h`, which
    reproduces the trajectories of a Vode network.

    NOTE: an ErrorVode cannot be clamped with `set("h", value)`: a clamped value is fixed while `u` changes, so it is
    not a fixed error. Use a Vode for clamped nodes (e.g. the output node during training).
    """

    def __init__(
        self,
        energy_fn: Callable[["Vode", RandomKeyGenerator], jax.Array] = se_energy,
        ruleset: dict = {},
        tforms: dict = {},
        param_type: type[VodeParam] = VodeParam,
        *param_args,
        **param_kwargs,
    ):
        """ErrorVode constructor. Arguments are as for `Vode`; the parameter created is the error `e`."""
        EnergyModule.__init__(self)

        self.e = param_type(*param_args, **param_kwargs)
        self.cache = param_type.Cache()
        self.energy_fn = static(energy_fn)
        self.ruleset = Ruleset(
            {STATUS.INIT: ("u <- u", "e <- u:zeros"), **ruleset},
            {"zeros": _zeros_like, **tforms},
        )
        self.shape = static(None)

    def _stored(self) -> VodeParam:
        return self.e

    def set(self, key: str, value: jax.Array | None, rkg: RandomKeyGenerator = RKG) -> "ErrorVode":
        if key == "h":
            raise ValueError(
                "An ErrorVode stores the error e = h - u, so 'h' cannot be set directly. Set 'e', or use a Vode for "
                "clamped nodes."
            )

        super().set(key, value, rkg)

        return self

    def get(self, key: str, default: Any | None = None, rkg: RandomKeyGenerator = RKG) -> jax.Array | Any | None:
        if key == "h":
            _e, _u = self.e.get(), super().get("u", rkg=rkg)
            if _e is None or _u is None:
                return default

            return jax.lax.stop_gradient(_u) + _e

        return super().get(key, default, rkg)

    def residual(self, rkg: RandomKeyGenerator = RKG) -> jax.Array:
        """`h - u` computed as `e + (sg(u) - u)`: its value is exactly `e`, and its gradient w.r.t. `u` is -1."""
        _u = cast(jax.Array, self.get("u", rkg=rkg))

        return self.e.get() + (jax.lax.stop_gradient(_u) - _u)


# Updates ##############################################################################################################


def apply_h_updates(
    fn: Callable,
    updates: PyTree,
    *args: Any,
    model: PyTree,
    order: int = 2,
    **kwargs: Any,
) -> None:
    """Applies state updates as steps in `h`, for a network that may contain ErrorVodes.

    For a Vode this is `h <- h + d`, the same as `Optim.apply_updates`. For an ErrorVode, `h_l` moving by `d_l` while
    the node below moves by `d_{l-1}` means

        e_l <- e_l + d_l - [u_l(h_{l-1} + d_{l-1}) - u_l(h_{l-1})].

    Computing the bracket as a difference of two forward passes would reintroduce the cancellation that ErrorVodes
    avoid, so it is computed as a Taylor expansion in forward mode: `J d` (order 1) plus `d^T H d / 2` (order 2). The
    fixed points are unaffected by the order; the trajectory differs from the Vode one by O(|d|^(order+1)) per step.

    Typical usage, replacing `optim_h.step(model, g)`:

    ```python
    updates = optim_h.step(model, g, apply_updates=False)
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        pxc.apply_h_updates(forward, updates, x, None, model=model)
    ```

    Args:
        fn (Callable): a function that runs the forward pass of `model` from its current states, writing each node's
            incoming activation `u` (e.g. the vmapped `forward` used for initialisation, without clamping).
        updates (PyTree): the updates returned by `Optim.step(..., apply_updates=False)`.
        *args, **kwargs: passed to `fn`.
        model (PyTree): the model, passed to `fn` as the keyword argument `model`.
        order (int, optional): order of the Taylor expansion of `u_l`, 1 or 2. Defaults to 2.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")

    # Pair every updated parameter with its update.
    _params: list[DynamicParam] = []
    _deltas: list[jax.Array] = []

    def _pair(u, p):
        if u is not None and (_u := u.get() if isinstance(u, BaseParam) else u) is not None:
            _params.append(p)
            _deltas.append(_u)

    jtu.tree_map(_pair, updates, model, is_leaf=lambda x: x is None or isinstance(x, BaseParam))

    _error_vodes: list[ErrorVode] = []
    tree_apply(lambda v: _error_vodes.append(v), lambda x: isinstance(x, ErrorVode), tree=model)

    _p0 = [p.get() for p in _params]

    if _error_vodes:
        # Every dynamic parameter of the model, so that the forward passes below leave no trace (e.g. jvp tracers
        # in the caches) behind them.
        _all = jtu.tree_leaves(model, is_leaf=lambda x: isinstance(x, DynamicParam))

        def _us(values: list[jax.Array]) -> list[jax.Array]:
            _snapshot = [dict(v) if isinstance(v, dict) else v for v in (p.get() for p in _all)]
            _key = RKG.key.get()
            try:
                for p, v in zip(_params, values, strict=True):
                    p.set(v)
                fn(*args, model=model, **kwargs)

                return [cast(jax.Array, v.get("u")) for v in _error_vodes]
            finally:
                for p, v in zip(_all, _snapshot, strict=True):
                    p.set(v)
                RKG.key.set(_key)

        def _du1(values: list[jax.Array]) -> list[jax.Array]:
            return jax.jvp(_us, (values,), (_deltas,))[1]

        if order == 1:
            _du = _du1(_p0)
        else:
            _first, _second = jax.jvp(_du1, (_p0,), (_deltas,))
            _du = [a + 0.5 * b for a, b in zip(_first, _second, strict=True)]

        _du_by_param = {id(v.e): du for v, du in zip(_error_vodes, _du, strict=True)}
    else:
        _du_by_param = {}

    for p, v, d in zip(_params, _p0, _deltas, strict=True):
        if id(p) in _du_by_param:
            p.set(v + d - _du_by_param[id(p)])
        else:
            p.set(v + d)
