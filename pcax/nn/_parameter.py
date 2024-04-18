__all__ = [
    "LayerParam",
    "LayerState",
]


from typing import Optional
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
    def __init__(
        self,
        value: Optional[jax.Array] = None
    ):
        super().__init__(value)


class LayerState(Param):
    def __init__(
        self,
        value: Optional[jax.Array] = None
    ):
        super().__init__(value)
