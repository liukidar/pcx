import equinox as eqx
from pcax.core.node import NODE_STATUS, NODE_TYPE

from ..structure.state import _State
from .decorators import batch_over, partials, with_grad


class Trainer:
    @staticmethod
    @batch_over(
        mask_kwargs={
            "model": lambda _, state: state.map_mask(
                lambda type: type == NODE_TYPE.X, "type"
            ),
            "x": True,
            "t": True,
        },
        mask_out=["model"],
    )
    def init_fn(state, model, x, t):
        with state.unfreeze(
            model, filter_fn=lambda _, type: type == NODE_TYPE.X, filter_args="type"
        ) as unforzen_model:
            unforzen_model.init(state, x, t)

            model = state.freeze(unforzen_model)

        return model

    @staticmethod
    @partials(
        {
            NODE_TYPE.X: {
                "grad_filter_fn": lambda _, type, status: type == NODE_TYPE.X
                and status != NODE_STATUS.FROZEN
            },
            NODE_TYPE.W: {
                "grad_filter_fn": lambda _, type, status: type == NODE_TYPE.W
                and status != NODE_STATUS.FROZEN
            },
            NODE_TYPE.X
            + NODE_TYPE.W: {
                "grad_filter_fn": lambda _, type, status: type == NODE_TYPE.X
                or type == NODE_TYPE.W
                and status != NODE_STATUS.FROZEN
            },
        }
    )
    def update_fn(
        state: _State,
        model,
        x_args=[],
        x_kwargs={},
        optim=None,
        loss_fn=None,
        loss_fn_args=[],
        loss_fn_kwargs={},
        grad_filter_fn=None,
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
            lambda state, model: {
                "model": state.partition(
                    model,
                    grad_filter_fn,
                    ["type", "status"],
                )
            },
        )
        def forward(
            state, model, x_args, x_kwargs, loss_fn, loss_fn_args, loss_fn_kwargs
        ):
            y = model(*x_args, **x_kwargs)
            loss = loss_fn(state, model, y, *loss_fn_args, **loss_fn_kwargs)

            return loss, y

        (loss, y), grads = forward(
            state, model, x_args, x_kwargs, loss_fn, loss_fn_args, loss_fn_kwargs
        )

        updates, optim_state = optim.update(
            [grads["model"]], *state.get_masks("optim"), [model]
        )
        state.save_mask("optim", optim_state, type="dynamic")

        return {
            "state": state,
            "model": eqx.apply_updates(model, updates[0]),
            "y": (y, loss),
        }
