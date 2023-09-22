__all__ = ["_RKGState", "RKG", "RandomKeyGenerator"]

from typing import Optional, List
import time

import jax

from .modules import Module
from .parameters import Param, reduce_none

########################################################################################################################
#
# RANDOM
#
# jax requires to keep track of the random generator state. This is done by passing a PRNGKey to every random function.
# pcax offers a random generator class that keeps track of the state and can be used to generate random keys. RKG is
# the default random generator used provided by pcax.
#
########################################################################################################################

# Utils ################################################################################################################


class _RKGState(Param):
    """RKGState is a state parameter that tracks a random generator state. It is meant to be used internally."""

    def __init__(self, seed: int):
        """_RKGState constructor.

        Args:
            seed: the initial seed of the random number generator.
        """
        super().__init__(jax.random.PRNGKey(seed), reduce_none)

    def seed(self, seed: int):
        """Sets a new random seed.

        Args:
            seed: the new initial seed of the random number generator.
        """
        self.value = jax.random.PRNGKey(seed)

    def split(self, n: int) -> List[jax.Array]:
        """Create multiple seeds from the current seed. This is used internally by pcax.Vectorize to ensure
        that random numbers are different in parallel computations.

        Args:
            n: the number of seeds to generate.
        """
        values = jax.random.split(self.value, n + 1)
        self._value = values[0]
        return values[1:]


# Random ###############################################################################################################


class RandomKeyGenerator(Module):
    """Random number generator module. Provide an imperative interface to generate random keys."""

    def __init__(self, seed: int = 0):
        """Create a random key generator, seed is the random generator initial seed."""
        super().__init__()
        self._key: Optional[_RKGState] = _RKGState(seed)

    def seed(self, seed: int = 0):
        """Sets a new random generator seed."""
        self._key.seed(seed)

    def __call__(self):
        """Generate a new key."""
        return self._key.split(1)[0]


"""Default random key generator. """
RKG = RandomKeyGenerator(time.time_ns())
