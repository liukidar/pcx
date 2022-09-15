import functools
from typing import Any, Callable
import jax
import equinox as eqx
from pcax.core.nn import NODE_STATUS
from pcaxrc.core.nn import NODE_TYPE

from pcaxrc.utils.functions import all_kwargs, call_kwargs


# def batch(
#     in_args_batch_fns: List[bool | Callable[[Any], bool]],
#     in_kwargs_batch_fns: Dict[str, bool | Callable[[Any], bool]] = {},
# ):
#     def decorator(fn):
#         @functools.wraps(fn)
#         def wrapper(*args, **kwargs):
#             in_axes_map = ()


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
                fn_kwargs[next(iter(fn_kwargs))] = eqx.combine(params, static)
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
    def init_fn(state, model, x):
        with state.unfreeze(
            model, filter_fn=lambda _, type: type == NODE_TYPE.X, filter_args="type"
        ) as unforzen_model:
            unforzen_model.init(state, x)

            model = state.freeze(unforzen_model)

        return model

    @staticmethod
    def update_fn(
        state,
        model,
        x_args=[],
        x_kwargs={},
        optim=None,
        loss_fn=None,
        loss_fn_args=[],
        loss_fn_kwargs={},
    ):
        if loss_fn is not None:

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

            updates, optim_state = optim.update([grad], state.masks["optim"], [model])
            state.masks["optim"] = optim_state
            eqx.apply_updates(model, updates[0]), (y, loss)
        else:
            y = model(*x_args, **x_kwargs)
            loss = None

        #

        return y, loss
