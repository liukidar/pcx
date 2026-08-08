__all__ = [
    "scan",
    "while_loop",
    "cond",
    "switch",
    "jit",
    "vmap",
    "value_and_grad",
]

from typing import Any, Hashable, Sequence, Callable

from ._transform import _BaseTransform, Jit, Vmap, ValueAndGrad
from ._flow import Scan, WhileLoop, Cond, Switch


# Flow ###############################################################################################################


def scan(
    f: _BaseTransform | Callable,
    xs: Sequence[Any] | None = None,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
) -> Scan:
    """Utility function to use the jax.lax.scan syntax for the :class:`~pcax.functional.Scan` transformation."""
    return Scan(f, xs=xs, length=length, reverse=reverse, unroll=unroll)


def while_loop(
    f: _BaseTransform | Callable,
    cond_fun: _BaseTransform | Callable,
) -> WhileLoop:
    """Utility function to use the jax.lax.while_loop syntax for the :class:`~pcax.functional.WhileLoop` transformation."""
    return WhileLoop(f, cond_fun=cond_fun)


def cond(
    true_fun: _BaseTransform | Callable,
    false_fun: _BaseTransform | Callable,
) -> Cond:
    """Utility function to use the jax.lax.cond syntax for the :class:`~pcax.functional.Cond` transformation."""
    return Cond(true_fun, false_fun)


def switch(
    branches: Sequence[_BaseTransform | Callable],
) -> Switch:
    """Utility function to use the jax.lax.switch syntax for the :class:`~pcax.functional.Switch` transformation."""
    return Switch(branches)


# Transform ############################################################################################################


def jit(
    static_argnums=None,
    # static_argnames=None,  # this is currently disabled as it hasn't been tested.
    donate_argnums=None,
    donate_argnames=None,
    **kwargs,
):
    def decorator(fn: _BaseTransform | Callable):
        return Jit(
            fn,
            static_argnums=static_argnums,
            # static_argnames=static_argnames,
            donate_argnums=donate_argnums,
            donate_argnames=donate_argnames,
            **kwargs,
        )

    return decorator


def vmap(
    kwargs_mask: Any = {},
    in_axes: Sequence[int | None] = (),
    out_axes: Sequence[int | None] = (),
    axis_name: str | None = None,
):
    def decorator(fn: _BaseTransform | Callable):
        return Vmap(
            fn, kwargs_mask, in_axes=in_axes, out_axes=out_axes, axis_name=axis_name
        )

    return decorator


def value_and_grad(
    kwargs_mask: Any = {},
    argnums: int | Sequence[int] = (),
    has_aux: bool = False,
    reduce_axes: Sequence[Hashable] = (),
):
    def decorator(fn: _BaseTransform | Callable):
        return ValueAndGrad(
            fn, kwargs_mask, argnums=argnums, has_aux=has_aux, reduce_axes=reduce_axes
        )

    return decorator
