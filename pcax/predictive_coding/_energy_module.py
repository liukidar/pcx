__all__ = []

from typing import Any, Callable, Type
import functools
from types import UnionType

import jax

from ..core._module import Module
from ..core._tree import tree_apply
from ..core._static import static


########################################################################################################################
#
# ENERGY MODULE
#
# predictive_coding.Module inherits from core.Module and adds a status attribute to the module to configure the behavior
# of its methods. It also introduces the concept of energy. In particular, it is used by 'preditive_coding.Vode'.
# A Predictive Coding Network should inherit from this class.
#
########################################################################################################################


# Core #################################################################################################################


class EnergyModule(Module):
    """Module inherits from core.Module. he extra '_status' attribute can be used to configure the behavior
    of its methods. In particular, it used by 'preditive_coding.Vode'.
    """
    def __init__(self) -> None:
        """Module constructor.
        """
        super().__init__()
        self._status = static(None)

    def energy(self) -> jax.Array:
        """Return the total energy of the module as the recursive sum of all the energies of its submodules.
        Note that differently from the Vodes, the energy is not cached.
        
        Returns:
            jax.Array: total energy of the module.
        """
        return functools.reduce(
            lambda x, y: x + y,
            (m.energy() for m in self.submodules(cls=EnergyModule))
        )
    
    def clear_params(self, filter: Callable[[Any], bool] | Type) -> None:
        """Set the selected parameters to None. This is especially useful to clear the cache of the parameters when needed.
        Note that, being pcax an imperative library, the change is done in-place and no updated module is returned.

        Args:
            filter (Callable[[Any], bool] | Type): filter function or type identifying the parameters to clear.
        """
        tree_apply(
            lambda p: p.set(None),
            filter if not isinstance(filter, type | UnionType) else lambda x: isinstance(x, filter),
            tree=self,
            recursive=False
        )

    @property
    def status(self) -> Any:
        return self._status.get()

    @status.setter
    def status(self, status: Any):
        self._status.set(status)
