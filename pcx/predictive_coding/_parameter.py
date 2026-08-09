__all__ = ["ParamCache", "VodeParam"]


import jax

from ..core._parameter import Param, ParamCache, ParamDict

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
        def __init__(self, params: dict[str, jax.Array] | None = None):
            super().__init__(params)

    def __init__(self, value: jax.Array | None = None):
        super().__init__(value)
