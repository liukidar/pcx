__all__ = [
    "ParamCache",
    "VodeParam"
]


from typing import Dict, Optional
import jax

from ..core._parameter import Param, ParamDict, ParamCache


########################################################################################################################
#
# PARAMETER
#
# We introduce different types of parameters to be used in the Vodes. This allow the user to distinguish them and target
# them with specify transformations.
#
########################################################################################################################


# Core #################################################################################################################


class VodeParam(Param):
    class Cache(ParamDict, ParamCache):
        def __init__(self, params: Dict[str, jax.Array] = None):
            super().__init__(params)

    def __init__(
        self,
        value: Optional[jax.Array] = None
    ):
        super().__init__(value)
