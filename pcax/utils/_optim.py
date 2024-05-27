__all__ = ["Optim"]

from jaxtyping import PyTree
import optax
import jax.tree_util as jtu
import equinox as eqx

from ..core._module import BaseModule
from ..core._parameter import Param, BaseParam, set, get
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

    def __init__(self, optax_opt: optax.GradientTransformation, parameters: PyTree | None = None):
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
        self, module: PyTree, grads: PyTree, scale_by_batch_size: bool = False, apply_updates: bool = True, mul: float = None
    ) -> None:
        """Performs a gradient update step similarly to Pytorch's 'optimizer.step()' by calling first 'optax_opt.update'
        and then 'eqx.apply_updates'.

        Args:
            module (PyTree): the module storing the target parameters.
            grads (PyTree): the computed gradients to apply. Provided gradients must match the same structure of the
                module used to initialise the optimizer.
        """

        # Filter out the Params that do not have a gradient (this, for example, includes all the StaticParam whose
        # current state my differ from the original parameter structure and thus be incomparible with the gradients
        # structure). By doing so, and by enforcing the basic requirement that the gradients for all target paramemters
        # are provided, we can safely assume that the filtered gradients structure matches the filtered target
        # parameters structure.
        # For example 'grads' could contain gradients computed for parameters not targeted by this optimiser without
        # causing any issue since they will be filtered out automatically.
        module = eqx.filter(module, self.filter.get(), is_leaf=lambda x: isinstance(x, BaseParam))

        if scale_by_batch_size is True:
            grads = jtu.tree_map(
                lambda x, f: x.set(x * x.shape[0]) if f is True else None,
                grads,
                self.filter.get(),
                is_leaf=lambda x: isinstance(x, BaseParam),
            )
        elif mul is not None:
            grads = jtu.tree_map(
                lambda x, f: x.set(x * mul) if f is True else None,
                grads,
                self.filter.get(),
                is_leaf=lambda x: isinstance(x, BaseParam),
            )
        else:
            grads = eqx.filter(grads, self.filter.get(), is_leaf=lambda x: isinstance(x, BaseParam))

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
        # he remaining parameters and allow them to change structure without affecting the functioning of the
        # optimizer.
        self.filter.set(
            jtu.tree_map(lambda x: get(x) is not None, parameters, is_leaf=lambda x: isinstance(x, BaseParam))
        )
        parameters = eqx.filter(parameters, self.filter.get(), is_leaf=lambda x: isinstance(x, BaseParam))

        self.state.set(self.optax_opt.init(parameters))

    def clear(self) -> None:
        self.state.set(None)
        self.filter.set(None)
