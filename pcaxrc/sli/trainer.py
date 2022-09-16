import functools
from typing import Any, Callable, Dict, List
import jax
import jax.tree_util as jtu
import equinox as eqx
from pcax.core.nn import NODE_STATUS
from pcaxrc.core.nn import NODE_TYPE

from pcaxrc.utils.functions import all_kwargs, call_kwargs
from ..lib.state import _State


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
                    **fn_kwargs
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


# TODO: specify grad_arg number
def with_grad(
    grad_filter_fn: Callable[[Any], bool],
    grad_callback_fn: Callable[[Any], None] = lambda model: model._clear_cache(),
):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            fn_kwargs = all_kwargs(fn, *args, **kwargs)

            def predict(params, static):
                # Update the first parameter passed to the function
                fn_kwargs["model"] = eqx.combine(params, static)
                loss, r = call_kwargs(fn, **fn_kwargs)
                call_kwargs(grad_callback_fn, **fn_kwargs)

                return loss, r

            params, static = call_kwargs(grad_filter_fn, **fn_kwargs)
            (loss, r), grad = jax.value_and_grad(predict, has_aux=True)(params, static)

            return (loss, r), grad

        return wrapper

    return decorator


class Trainer:
    @staticmethod
    @batch_over(
        mask_kwargs={
            "model": lambda _, state: state.map_mask(
                lambda type: type == NODE_TYPE.X, "type"
            ),
            "x": True,
        },
        mask_out=["model"],
    )
    def init_fn(state, model, x):
        with state.unfreeze(
            model, filter_fn=lambda _, type: type == NODE_TYPE.X, filter_args="type"
        ) as unforzen_model:
            unforzen_model.init(state, x)

            model = state.freeze(unforzen_model)

        return model

    @staticmethod
    def update_fn(
        state: _State,
        model,
        x_args=[],
        x_kwargs={},
        optim=None,
        loss_fn=None,
        loss_fn_args=[],
        loss_fn_kwargs={},
    ):
        @batch_over(
            mask_kwargs={
                "model": lambda _, state: state.map_mask(
                    lambda type: type == NODE_TYPE.X, "type"
                ),
                "x_args": True,
                "loss_fn_args": True,
            },
            mask_out=[True, True],
        )
        @with_grad(
            lambda state, model: state.partition(
                model,
                lambda _, type, status: type == NODE_TYPE.X
                and status != NODE_STATUS.FROZEN,
                ["type", "status"],
            ),
        )
        def forward(
            state, model, x_args, x_kwargs, loss_fn, loss_fn_args, loss_fn_kwargs
        ):
            y = model(*x_args, **x_kwargs)
            loss = loss_fn(state, model, y, *loss_fn_args, **loss_fn_kwargs)

            return loss, y

        (loss, y), grad = forward(
            state, model, x_args, x_kwargs, loss_fn, loss_fn_args, loss_fn_kwargs
        )

        updates, optim_state = optim.update([grad], *state.get_masks("optim"), [model])
        state.save_mask("optim", optim_state)

        return state, eqx.apply_updates(model, updates[0]), y, loss
