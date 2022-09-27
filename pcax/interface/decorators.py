import functools
from logging import warning
from typing import Any, Callable, Dict, List
import jax
import jax.tree_util as jtu
import equinox as eqx
from ..utils.functions import all_kwargs, call_kwargs, ensure_tuple
from contextlib import contextmanager
from ..core.environment import _C


@contextmanager
def debug():
    prev = _C["debug"]
    _C["debug"] = True

    try:
        yield None
    finally:
        _C["debug"] = prev


@contextmanager
def force_forward():
    prev = _C["force_forward"]
    _C["force_forward"] = True

    try:
        yield None
    finally:
        _C["force_forward"] = prev


def batch_over(
    mask_kwargs: Dict[str, bool | Callable[[Any], bool]],
    mask_out: List[str | bool | Callable[[Any], bool]],
    mask_fn: Callable[[Any], bool] = lambda _: False,
    axis_name: str = "AX_BATCH",
    out_as_tuple: bool = False,
):
    def get_batch_mask(
        mask_dict: Dict[str, bool | Callable[[Any], bool]], param: bool | str
    ) -> Callable[[Any], bool]:
        mask = param
        if isinstance(mask, str):
            mask = mask_dict.get(mask, mask_fn)

        if not callable(mask):
            return lambda _: mask
        else:
            return mask

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            fn_kwargs, param_names = all_kwargs(
                fn, *args, **kwargs, get_params_names=True
            )

            in_axes_map = tuple(
                call_kwargs(
                    get_batch_mask(mask_kwargs, param_name),
                    fn_kwargs[param_name],
                    **fn_kwargs,
                )
                for param_name in param_names
            )

            out_axes_map = tuple(
                call_kwargs(get_batch_mask(mask_kwargs, param), param, **fn_kwargs)
                for param in mask_out
            )
            if len(out_axes_map) == 1 and out_as_tuple is False:
                out_axes_map = out_axes_map[0]

            return jax.vmap(
                fn,
                in_axes=jtu.tree_map(lambda v: 0 if v else None, in_axes_map),
                out_axes=jtu.tree_map(lambda v: 0 if v else None, out_axes_map),
                axis_name=axis_name,
            )(*(fn_kwargs[param_name] for param_name in param_names))

        return wrapper

    return decorator


def with_grad(
    grad_filter_fn: Callable[[Any], bool],
    grad_callback_fn: Callable[[Any], None] = lambda model: model._clear_cache(),
):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            fn_kwargs = all_kwargs(fn, *args, **kwargs)

            params_and_static = call_kwargs(grad_filter_fn, **fn_kwargs)

            def predict(params, static):
                # Update the first parameter passed to the function
                fn_kwargs.update(
                    {
                        k: eqx.combine(p, s)
                        for k, p, s in zip(params_and_static.keys(), params, static)
                    }
                )
                loss, r = call_kwargs(fn, **fn_kwargs)
                call_kwargs(grad_callback_fn, **fn_kwargs)

                return loss, r

            (loss, r), grads = jax.value_and_grad(predict, has_aux=True)(
                *tuple(zip(*params_and_static.values()))
            )

            return (loss, r), {
                name: grad for name, grad in zip(params_and_static, grads)
            }

        return wrapper

    return decorator


class __partials_getter:
    def __init__(self, partial_fns):
        self.partial_fns = partial_fns

    def __getitem__(self, keys):
        keys = ensure_tuple(keys)

        return (
            tuple(self.partial_fns[key] for key in keys)
            if len(keys) > 1
            else self.partial_fns[keys[0]]
        )


def partials(
    static_kwargs: Dict[Any, Any],
):
    def decorator(fn):
        def make_wrapper(static_kwargs):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **{**static_kwargs, **kwargs})

            return wrapper

        return __partials_getter(
            {name: make_wrapper(static_kwargs[name]) for name in static_kwargs}
        )

    return decorator


__jit_debug = {
    "hash": {},
    "counter": {},
}


def jit(fn):
    def body(_hash, *args, **kwargs):
        if _C["debug"] is True:
            if fn not in __jit_debug["counter"]:
                __jit_debug["counter"][fn] = 0
            __jit_debug["counter"][fn] += 1

            print(
                f"[DEBUG] Function '{fn.__name__}' has been compiled #{__jit_debug['counter'][fn]} times"
                f" (hash: {_hash})."
            )

        return fn(*args, **kwargs)

    def decorator(device=None, *, _hash: Any | None = None, **fixed_kwargs):
        if _C["debug"] is True:

            @functools.wraps(fn)
            def debug_wrapper(*args, **kwargs):
                (_, static_args) = eqx.partition(args, lambda _: True)
                (_, static_kwargs) = eqx.partition(kwargs, lambda _: True)

                debug_hash = hash(static_args) + hash(tuple(static_kwargs.items()))

                if (fn, _hash) in __jit_debug["hash"]:
                    if debug_hash not in __jit_debug["hash"][(fn, _hash)]:
                        warning(
                            f"Function '{fn.__name__}' has been called with new static arguments"
                            " without recompiling the function with a new hash."
                            " Calling the function with new arguments"
                            " does not trigger a recompilation unless a new hash is specified."
                            " Without recompilation a previously cached version of the function will be using,"
                            " ignoring the new static arguments. This may be the desired behavior."
                            " If not, please specify a new hash for the function using the _hash keyword."
                        )
                        __jit_debug["hash"][(fn, _hash)].append(debug_hash)
                else:
                    __jit_debug["hash"][(fn, _hash)] = [debug_hash]

                return jax.jit(
                    body,
                    static_argnums=(0),
                    static_argnames=fixed_kwargs.keys(),
                    device=device,
                )(_hash, *args, **kwargs, **fixed_kwargs)

            return debug_wrapper
        else:

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return jax.jit(
                    body,
                    static_argnums=(0),
                    static_argnames=fixed_kwargs.keys(),
                    device=device,
                )(_hash, *args, **kwargs, **fixed_kwargs)

            return wrapper

    return decorator
