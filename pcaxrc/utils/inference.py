import functools
from typing import Any, Callable, List
import jax
import equinox as eqx

from pcax.core.nn import NodeInfo


def compute_grad():
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state, model, *args, grad_filter_fn, **kwargs):
            def predict(params, static):
                model = eqx.combine(params, static)
                loss, r = fn(state, model, *args, **kwargs)
                model._clear_cache()

                return loss, r

            params, static = state.partition(model, grad_filter_fn)
            (loss, r), grad = jax.value_and_grad(predict, has_aux=True)(params, static)

            return (loss, r), grad

        return wrapper

    return decorator


def batch_over(
    batch_in_mask: List[Any] | None = None,
    batch_out_mask: List[Any] | None = None,
    batch_filter_fn: Callable[[NodeInfo], bool] | None = None,
    batch_in_full_mask: List[Any] | None = None,
    batch_out_full_mask: List[Any] | None = None,
):
    assert (
        batch_in_mask is None or batch_in_full_mask is None
    ), "You can't specify both batch_in_mask and batch_in_full_mask"
    assert (
        batch_out_mask is None or batch_out_full_mask is None
    ), "You can't specify both batch_out_mask and batch_out_full_mask"
    assert (
        batch_filter_fn is None or batch_in_full_mask is None
    ), "You can't specify both batch_filter_fn and batch_in_full_mask"
    assert (
        batch_filter_fn is not None or batch_out_full_mask is not None
    ), "You need to specify a batch_filter_fn if you don't specify a batch_out_full_mask"

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state, model, *args, _batch_fn: bool = True):
            if _batch_fn:
                if batch_in_full_mask is not None:
                    in_axes_map = batch_in_full_mask
                else:
                    mask = batch_in_mask or []
                    in_axes_map = (
                        None,
                        state.model_vmap_mask(batch_filter_fn),
                        *mask,
                        *([None] * (len(args) - len(mask))),
                    )

                if batch_out_full_mask is not None:
                    out_axes_map = batch_out_full_mask
                else:
                    mask = batch_out_mask or []
                    out_axes_map = (*in_axes_map[:2], *mask)

                return jax.vmap(
                    fn, in_axes=in_axes_map, out_axes=out_axes_map, axis_name="__batch"
                )(state, model, *args)
            else:
                return fn(state, model, *args)

        return wrapper

    return decorator
