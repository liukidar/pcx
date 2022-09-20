import importlib as _importlib

# Simple Library Interface

from .state import (
    DefaultState,
)

from .trainer import (
    Trainer,
)

from .decorators import (
    batch_over,
    with_grad,
    partials,
    jit,
)

__all__ = [
    # .state
    "DefaultState",
    "Trainer",
    # .decorators
    "batch_over",
    "with_grad",
    "partials",
    "jit",
]


def __dir__():
    return __all__


module_name = "pcax.sli"
submodules = ["flow", "optim", "state"]


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f"{module_name}.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module '{module_name}' has no attribute '{name}'")
