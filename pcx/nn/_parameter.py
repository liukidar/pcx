__all__ = [
    "LayerParam",
    "LayerState",
]


import jax

from ..core._parameter import Param

########################################################################################################################
#
# PARAMETER
#
# We introduce different types of parameters to be used in the layers. This allow the user to distinguish between them.
#
########################################################################################################################


# Core #################################################################################################################


class LayerParam(Param):
    def __init__(self, value: jax.Array | None = None):
        super().__init__(value)


class LayerState(Param):
    def __init__(self, value: jax.Array | None = None):
        super().__init__(value)
