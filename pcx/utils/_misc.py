import contextlib
from collections.abc import Callable
from typing import Any

from jaxtyping import PyTree

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
    status: str | tuple | None = None,
    *,
    clear_params: Callable[[Any], bool] | type | tuple | None = None,
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
        status (str | tuple | None, optional): the status to apply to the module and its submodules. On exit, each
            module's status is restored to what it was on entry. If a tuple is provided, the first element is used
            for the duration of the step and the second element is applied on exit instead of restoring.
        clear_params (Callable[[Any], bool] | Type | Tuple, optional): Target parameters to clear. The value is
            directly passed to the 'EnergyModule.clear_params', so refer to that method for more information.
            If a tuple is provided, the first element is used to call '.clear_params' before the step and the second
            element is used after. If a single element is provided, it is used AFTER the step and no clearing happens
            before that.
    """

    # Enforce status to be a tuple.
    status = (status,) if not isinstance(status, tuple) else status

    clear_params = (None, clear_params) if not isinstance(clear_params, list | tuple) else clear_params

    if clear_params[0] is not None:
        module.clear_params(clear_params[0])

    # Record each module's status on entry so a scalar status is restored on exit. Modules visited more than
    # once (see `tree_apply`) are recorded once per visit; restoring in reverse order leaves the entry value.
    previous: list[tuple[EnergyModule, Any]] = []

    def _enter(m: EnergyModule) -> None:
        previous.append((m, m.status))
        m._status.set(status[0])

    tree_apply(_enter, lambda x: isinstance(x, EnergyModule), tree=module)

    try:
        yield
    finally:
        if clear_params[1] is not None:
            module.clear_params(clear_params[1])

        if len(status) > 1:
            tree_apply(
                lambda m: m._status.set(status[1]),
                lambda x: isinstance(x, EnergyModule),
                tree=module,
            )
        else:
            for m, s in reversed(previous):
                m._status.set(s)
