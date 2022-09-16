import functools
from typing import Any, Callable, Dict, List
import jax
import jax.tree_util as jtu
import equinox as eqx
from ..utils.functions import all_kwargs, call_kwargs, ensure_tuple


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
    class __partial_evaluator:
        def __init__(self, partial_fns):
            self.partial_fns = tuple(partial_fns)

        def __call__(self, **kwargs):
            return tuple(
                functools.partial(partial_fn, **kwargs)
                for partial_fn in self.partial_fns
            )

    def __init__(self, partial_fns):
        self.partial_fns = partial_fns

    def __getitem__(self, keys):
        keys = ensure_tuple(keys)
        return self.__partial_evaluator(self.partial_fns[key] for key in keys)


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


__jit_counter = {}


def jit(show_jit_count: bool = False, **static_kwargs):
    def decorator(fn):
        def fn_with_counter(*args, **kwargs):
            if fn not in __jit_counter:
                __jit_counter[fn] = 0
            __jit_counter[fn] += 1

            if show_jit_count:
                print(
                    f"[DEBUG] Function '{fn.__name__}' has been compiled #{__jit_counter[fn]} times"
                )
            return fn(*args, **kwargs)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return jax.jit(fn_with_counter, static_argnames=static_kwargs.keys())(
                *args, **{**static_kwargs, **kwargs}
            )

        return wrapper

    return decorator
