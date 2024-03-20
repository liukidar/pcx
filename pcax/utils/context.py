from typing import Any, Callable, Type
import contextlib

from ..core._tree import tree_apply
from ..predictive_coding._energy_module import EnergyModule


@contextlib.contextmanager
def step(
    module: EnergyModule,
    status: str | None = None,
    *,
    clear_params: Callable[[Any], bool] | Type = None
):
    tree_apply(lambda m: m._status.set(status), lambda x: isinstance(x, EnergyModule), tree=module)

    yield
    
    tree_apply(lambda m: m._status.set(None), lambda x: isinstance(x, EnergyModule), tree=module)
    
    if clear_params is not None:
        module.clear_params(clear_params)
