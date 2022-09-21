from typing import Any, Callable, List
import jax.tree_util as jtu
import jax.numpy as jnp
import optax
import equinox as eqx
from pcax.core.node import NodeInfo
from pcax.structure.state import _State, Param, StateAction, is_param


####################################################################################################
# Actions
####################################################################################################


# Creates the default masks (i.e., 'type' and 'status') for the given model.
class SACreateDefaultMasks(StateAction):
    def __init__(self, **kwargs):
        super().__init__("create_default_masks", **kwargs)

    def __call__(
        self,
        state: _State,
        model: Any,
        param_filter: Callable[[Any], bool] = is_param,
        **kwargs
    ) -> None:
        # Create the default model mask, containing the 'node_info' for each parameter
        def mask_fn(node, state, is_leaf):
            if hasattr(node, "_node_info"):
                state = getattr(node, "_node_info")

            # If node is a leaf, check if it is filtered out
            if (
                is_leaf
                and param_filter is not None
                and param_filter(node, state) is False
            ):
                state = None

            return state

        type_info_mask = state.create_mask(model, mask_fn, root=NodeInfo())
        state.save_mask(
            "type", jtu.tree_map(lambda node_info: node_info.type, type_info_mask)
        )
        state.save_mask(
            "status", jtu.tree_map(lambda node_info: node_info.status, type_info_mask)
        )

        return {"model": model}


# Transforms all the parameters inside the model in naked values.
# Normally is called after the info stored inside the parameters have been saved to other masks.
class SAStandardiseModelParameters(StateAction):
    def __init__(self, **kwargs):
        super().__init__("standardise_model_parameters", **kwargs)

    def __call__(self, state: _State, model: Any, **kwargs):
        # Replace every parameter in the model with a Param object to have a consistent structure
        # then replace all of them with their naked value.
        def where(model):
            return jtu.tree_leaves(
                jtu.tree_map(lambda _, value: value, *state.get_masks("type"), model)
            )

        model = eqx.tree_at(where, model, replace_fn=lambda value: Param(value).data)

        return {"model": model}


# Initialize the model Xs with a default empty value, keeping track of the given batch size.
# This is needed to correctly map the model with vmap.
# The initialization relies on trainer's 'init_fn' function.
# After the model initialization, an optimizer is created and initialized with the model parameters.
#
# If a new batch size or optimizer is necessary, this action must be called again with the new values.
class SAInit(StateAction):
    def __init__(self, **kwargs):
        super().__init__("init", **kwargs)

    def __call__(
        self,
        state: _State,
        model: Any,
        batch_size: int,
        input_shape,
        optim_fn: Callable[[_State], optax.GradientTransformation] = None,
        trainer=None,
        init_fn_args: List[Any] = (),
        **kwargs
    ):
        with state.unfreeze(model) as unforzen_model:
            # Dummy init: fill the PC layers with dummy empty arrays so they can be correctly mapped by vmap
            x = jnp.empty(shape=(batch_size, 0))
            unforzen_model._dummy_init(x)
            model = state.freeze(unforzen_model)

        if trainer is not None:
            model = trainer.init_fn(
                state, model, jnp.empty(shape=(batch_size, *input_shape)), *init_fn_args
            )

        r_kwargs = {"model": model}

        if optim_fn is not None:
            assert (
                trainer is not None
            ), "Trainer must be provided to create the optimizer"
            # Create optimizer now that both neurons and weights are initialized with the correct shape
            optim = optim_fn(state)
            optim_mask = optim.init(
                [state.map_mask(lambda _, param: param, ["type", model])]
            )
            state.save_mask("optim", optim_mask, type="dynamic")
            r_kwargs["optim"] = optim

        return r_kwargs
