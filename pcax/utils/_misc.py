from typing import Any, Callable, Type, Tuple
from jaxtyping import PyTree
import contextlib

from ..core._tree import tree_apply
from ..predictive_coding._energy_module import EnergyModule


########################################################################################################################
#
# MISC
#
########################################################################################################################


@contextlib.contextmanager
def step(
    module: EnergyModule | PyTree,
    status: str | None | Tuple = None,
    *,
    clear_params: Callable[[Any], bool] | Type | Tuple = None,
):
    """Applies common operations to a model before and after a step (normally a weight and/or state update).
    It is useful as settings the model's status and clearing the parameters cache allows to control the model's
    behavior. Example of usage:

    ```python
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        # compute energy and gradients
        # apply gradients
    ```

    Args:
        module (EnergyModule | PyTree): the target module.
        status (str | None, optional): the status to apply to the module and its submodules.
        clear_params (Callable[[Any], bool] | Type | Tuple, optional): Target parameters to clear. The value is
            directly passed to the 'EnergyModule.clear_params', so refer to that method for more information.
            If a tuple is provided, the first element is used to call '.clear_params' before the step and the second
            element is used after. If a single element is provided, it is used AFTER the step and no clearing happens
            before that.
    """

    # Enforce status to be a tuple.
    clear_params = (
        (None, clear_params)
        if not isinstance(clear_params, list | tuple)
        else clear_params
    )

    if clear_params[0] is not None:
        module.clear_params(clear_params[0])

    tree_apply(
        lambda m: m._status.set(status),
        lambda x: isinstance(x, EnergyModule),
        tree=module,
    )

    yield

    if clear_params[1] is not None:
        module.clear_params(clear_params[1])
