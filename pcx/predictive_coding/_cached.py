__all__ = [
    "Cached",
    "LayerCache",
]

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.custom_derivatives import SymbolicZero

from ..core._parameter import BaseParam, ParamCache, ParamDict
from ._energy_module import EnergyModule
from ._vode import STATUS

########################################################################################################################
#
# CACHED LAYER
#
# A wrapper for a *linear* layer (Linear, Conv, ConvTranspose, ...) that keeps the layer's output z = W h + b between
# inference steps instead of recomputing it from h. During the forward initialisation (status INIT) the layer runs
# normally and z is stored; afterwards the stored z is returned through a custom JVP whose tangent rule applies the
# layer to the tangents, W dh + dW h + db: the derivative of a linear layer is the layer itself, so no gradient is
# written by hand and JAX transposes the rule for reverse mode. Consequences:
#
#   - the forward pass of an inference step costs no matmul; the gradient pass costs the same W^T(.) as a plain layer;
#   - `apply_h_updates` moves the states by d and obtains dz = W d from the same tangent rule, which it uses both to
#     update the error nodes above (Taylor step of f(z)) and to update the cache, z <- z + dz. One matmul per layer;
#   - together with ErrorVodes, no quantity of size |h| is ever subtracted from another: h = f(z) + e only appears
#     as an input to matmuls (weight gradient), like an activation in backprop. The errors may then be stored in
#     low precision (bf16) while z keeps a float32 master copy.
#
# The cache is a `LayerCache` parameter, distinct from `VodeParam.Cache` so that clearing the Vode caches at every
# step leaves it in place. It must be included in the vmap mask of batched functions
# (e.g. `pxu.M(pxc.VodeParam | pxc.VodeParam.Cache | pxc.LayerCache).to((None, 0))`) and cleared, like the Vode
# states, before a forward pass with a different batch size (`model.clear_params(pxc.LayerCache)`).
#
# The cache is only valid if every change of the layer's input goes through `apply_h_updates` (or a new INIT forward).
# Stepping the states with `optim.step` directly leaves it stale.
#
########################################################################################################################


# Core #################################################################################################################


class LayerCache(ParamDict, ParamCache):
    """Cache of a `Cached` layer: its output `z` at the current states."""

    def __init__(self, params: dict[str, jax.Array] | None = None):
        super().__init__(params)


def _values(tree: Any) -> Any:
    return jtu.tree_map(
        lambda w: w.get() if isinstance(w, BaseParam) else w,
        tree,
        is_leaf=lambda w: isinstance(w, BaseParam),
    )


def _no_bias(arrays: Any) -> Any:
    """The layer's arrays with every `bias` leaf zeroed: W dh must not add b."""
    return jtu.tree_map_with_path(
        lambda path, a: jnp.zeros_like(a) if path and getattr(path[-1], "name", None) == "bias" else a,
        arrays,
    )


def _is_zero(tangent: Any) -> bool:
    return all(isinstance(t, SymbolicZero) for t in jtu.tree_leaves(tangent))


def _cached_apply(nn: Any, x: jax.Array, z: jax.Array, key: Any) -> jax.Array:
    """Return `z` with the derivatives of `nn(x)`. `nn` is the (equinox) layer with concrete arrays."""
    arrays, static_nn = eqx.partition(nn, eqx.is_array)

    def apply(arrays_: Any, x_: jax.Array) -> jax.Array:
        return eqx.combine(arrays_, static_nn)(x_, key=key)

    @jax.custom_jvp
    def cached(arrays_: Any, x_: jax.Array, z_: jax.Array) -> jax.Array:
        return z_

    def cached_jvp(
        primals: tuple[Any, jax.Array, jax.Array], tangents: tuple[Any, Any, Any]
    ) -> tuple[jax.Array, jax.Array]:
        arrays_, x_, z_ = primals
        darrays, dx, _dz = tangents
        tangent = None
        if not isinstance(dx, SymbolicZero):
            tangent = apply(_no_bias(arrays_), dx)
        if not _is_zero(darrays):
            darrays = jtu.tree_map(
                lambda t, a: jnp.zeros_like(a) if isinstance(t, SymbolicZero) else t,
                darrays,
                arrays_,
                is_leaf=lambda t: isinstance(t, SymbolicZero),
            )
            t_w = apply(darrays, x_)
            tangent = t_w if tangent is None else tangent + t_w
        if tangent is None:
            tangent = jnp.zeros_like(z_)

        # The primal goes back through `cached` so that an enclosing differentiation (nested jvp in
        # `apply_h_updates`, order 2) applies this rule again and sees the dependence on x; a bare `z_` would be a
        # constant to it and the second-order term would vanish.
        return cached(arrays_, x_, z_), tangent

    cached.defjvp(cached_jvp, symbolic_zeros=True)

    return cached(arrays, x, z)


class Cached(EnergyModule):
    """
    Wraps a linear layer so that its output is cached between inference steps (see the module docstring).

    ```python
    self.layers = [pxc.Cached(pxnn.Linear(d_in, d_out)) for ...]
    ```

    The wrapped layer must be linear in its input and in its parameters (Linear, Conv, ConvTranspose, average
    pooling, ...). Status `STATUS.INIT`, or an empty cache, runs the layer and fills the cache.
    """

    def __init__(self, layer: Callable[..., jax.Array]):
        super().__init__()
        self.layer = layer
        self.cache = LayerCache()

    def __call__(self, x: jax.Array, key: Any = None) -> jax.Array:
        _cache = self.cache.get()
        _z = _cache.get("z") if _cache is not None else None

        if self.status == STATUS.INIT or _z is None:
            _z = self.layer(x, key=key)
        else:
            _z = _cached_apply(_values(self.layer.nn), x, _z, key)

        self.cache["z"] = _z

        return _z
