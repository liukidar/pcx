__all__ = ["Optim"]

from typing import Callable, Any
from jaxtyping import PyTree
import optax
import jax.tree_util as jtu
import equinox as eqx

from ..core._module import BaseModule
from ..core._parameter import Param, DynamicParam, BaseParam, set, get
from ..core._static import static


########################################################################################################################
#
# OPTIM
#
# Optim offers a simple interface to the optax library. Being a 'BaseModule' it can be pass through pcax transformations
# with its state being tracked and updated. Note that init can be called anytime to reset the optimizer state. This
# can be helpful when the optimizer is used in a loop and the state needs to be reset at each iteration (for example,
# this may be the case for the vode opimitzer after each mini-batch).
#
# DEV NOTE: currently all the state is stored as a single parameter. This may be insufficient for advanced learning
# techniques that require, for example, differentiating with respect to some of the optimizer's values.
# In such case, this optimizer class should be upgradable to the same design pattern used for 'Layers', which substitues
# each individual weights with a different parameter (which when '.update' is called, can be firstly replaced with their
# values, similarly to the 'Layer.__call__' method).
#
########################################################################################################################


class Optim(BaseModule):
    """Optim inherits from core.BaseModule and thus it is a pytree. It is a thin wrapper around the optax library."""

    def __init__(
        self, optax_opt: optax.GradientTransformation, parameters: PyTree | None = None
    ):
        """Optim constructor.

        Args:
            optax_opt (optax.GradientTransformation): the optax constructor function.
            parameters (PyTree | None, optional): target parameters. The init method can be called separately by passing
                None.
        """
        self.optax_opt = static(optax_opt)
        self.state = Param(None)
        self.filter = static(None)

        if parameters is not None:
            self.init(parameters)

    def step(
        self,
        module: PyTree,
        grads: PyTree,
        scale_by: float | None = None,
        apply_updates: bool = True,
        allow_none: bool = False,
    ) -> PyTree:
        """Performs a gradient update step similarly to Pytorch's 'optimizer.step()' by calling first 'optax_opt.update'
        and then 'eqx.apply_updates'.

        Args:
            module (PyTree): the module storing the target parameters.
            grads (PyTree): the computed gradients to apply. Provided gradients must match the same structure of the
                module used to initialise the optimizer.
            scale_by (float, optional): if given, the gradients are multiplied by value before calling the optimizer.
            apply_updates (bool, optional): if True, the updates are applied to the module parameters, if False, they
                are simply returned.
            allow_none (bool, optional): if True, the method will not raise an error if some gradients are None.

        Returns:
            PyTree: returns the computed updates.
        """

        # Filter out the Params that do not have a gradient (this, for example, includes all the StaticParam whose
        # current state my differ from the original parameter structure and thus be incomparible with the gradients
        # structure). By doing so, and by enforcing the basic requirement that the gradients for all target paramemters
        # are provided, we can safely assume that the filtered gradients structure matches the filtered target
        # parameters structure.
        # For example 'grads' could contain gradients computed for parameters not targeted by this optimiser without
        # causing any issue since they will be filtered out automatically.

        _is_valid_grads = True

        if scale_by is not None:

            def _map_grad(_, g):
                nonlocal _is_valid_grads
                if get(g) is None:
                    _is_valid_grads = False

                    return g

                return set(g, g * scale_by)
        else:

            def _map_grad(_, g):
                nonlocal _is_valid_grads
                if get(g) is None:
                    _is_valid_grads = False

                return g

        grads = jtu.tree_map(
            _map_grad,
            self.filter.get(),
            grads,
            is_leaf=lambda x: isinstance(x, BaseParam),
        )

        if _is_valid_grads is False:
            if allow_none is False:
                raise ValueError("Gradients for some parameters are None.")
            return None

        module = jtu.tree_map(
            lambda _, x: x,
            self.filter.get(),
            module,
            is_leaf=lambda x: isinstance(x, BaseParam),
        )

        updates, state = self.optax_opt.update(
            grads,
            self.state.get(),
            module,
        )
        self.state.set(state)

        if apply_updates:
            self.apply_updates(module, updates)

        return updates

    def apply_updates(self, module: PyTree, updates: PyTree) -> None:
        """Applies the updates to the module parameters.

        Args:
            module (PyTree): the module storing the target parameters.
            updates (PyTree): the updates to apply. Provided updates must match the same structure of the module used to
                initialise the optimizer.
        """
        jtu.tree_map(
            lambda u, p: set(p, eqx.apply_updates(get(p), get(u))),
            updates,
            module,
            is_leaf=lambda x: isinstance(x, BaseParam),
        )

    def init(self, parameters: PyTree) -> None:
        # We compute a static filter identifying the parameters given to be optimised. This is useful to filter out
        # the remaining parameters and allow them to change structure without affecting the functioning of the
        # optimizer.
        self.filter.set(
            jtu.tree_map(
                lambda _: True,
                parameters,
                is_leaf=lambda x: isinstance(x, BaseParam),
            )
        )

        self.state.set(self.optax_opt.init(parameters))

    def clear(self) -> None:
        """Reset the optimizer state.
        """
        self.state.set(None)
        self.filter.set(None)


class OptimTree(BaseModule):
    """OptimTree creates multiple optimizers for each leaf of the provided parameters, specified by `leaf_fn`. This is useful when
    different set of parameters are optimized at separate times. By default, a different optimizer is created for each Param.
    """

    def __init__(
        self,
        optax_opt: optax.GradientTransformation,
        leaf_fn: Callable[[Any], bool],
        parameters: PyTree | None = None,
    ):
        """OptimTree constructor.

        Args:
            optax_opt (optax.GradientTransformation): the optax constructor function.
            leaf_fn (Callable[[Any], bool]): function to specify which nodes to target for optimization. For each node a separate
                optimizer is created.
            parameters (PyTree | None, optional): target parameters. The init method can be called separately by passing
                None.
        """

        self.optax_opt = static(optax_opt)
        self.leaf_fn = static(leaf_fn)
        self.state = Param(None)

        if parameters is not None:
            self.init(parameters)

    def init(self, parameters: PyTree) -> None:
        leaves, structure = jtu.tree_flatten(
            parameters, is_leaf=lambda x: isinstance(x, DynamicParam) or self.leaf_fn(x)
        )

        optims = (Optim(self.optax_opt, n) for n in leaves)

        self.state.set(jtu.tree_unflatten(structure, optims))

    def step(
        self,
        module: PyTree,
        grads: PyTree,
        scale_by: float | None = None,
        apply_updates: bool = True,
    ) -> PyTree:
        """Performs a gradient update step similarly to Pytorch's 'optimizer.step()' by calling first 'optax_opt.update'
        and then 'eqx.apply_updates'.

        Args:
            module (PyTree): the module storing the target parameters.
            grads (PyTree): the computed gradients to apply. Provided gradients must match the same structure of the
                module used to initialise the optimizer. If the a gradient is None, the corresponding parameter group
                (i.e., the set of parameters contained in a node specified by the constructor's 'leaf_fn') is skipped
                during optimization.
            scale_by (float, optional): if given, the gradients are multiplied by value before calling the optimizer.
            apply_updates (bool, optional): if True, the updates are applied to the module parameters, if False, they
                are simply returned.
        Returns:
            PyTree: returns the computed updates.
        """

        # Each optimizer independently checks if the gradients are None and skips the optimization step if so.
        updates = jtu.tree_map(
            lambda optim, g, m: optim.step(
                m, g, scale_by=scale_by, apply_updates=apply_updates, allow_none=True
            ),
            self.state.get(),
            grads,
            module,
            is_leaf=lambda x: isinstance(x, Optim),
        )

        return updates

    def apply_updates(self, module: PyTree, updates: PyTree) -> None:
        # TODO: check this works properly as intended
        jtu.tree_map(
            lambda optim, u, m: optim.apply_updates(m, u),
            self.state.get(),
            updates,
            module,
            is_leaf=lambda x: isinstance(x, Optim),
        )

    def clear(self) -> None:
        """Reset each optimizer state.
        """

        self.state.set(None)
