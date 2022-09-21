from typing import List
from pcax.structure.state import _State, StateAction
from .actions import SACreateDefaultMasks, SAStandardiseModelParameters, SAInit


####################################################################################################
# States
####################################################################################################


# Inherits from State and defines a custom initialization function
class DefaultState(_State):
    def __init__(
        self,
        actions: List[StateAction] = [
            SACreateDefaultMasks(),
            SAStandardiseModelParameters(),
            SAInit(),
        ],
    ) -> None:
        super().__init__(actions)
