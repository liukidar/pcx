__all__ = [
    "RKGState",
    "RKG",
    "RandomKeyGenerator"
]

from typing import Tuple
from jaxtyping import ArrayLike
import time

import jax

from ._parameter import Param
from ._module import BaseModule


########################################################################################################################
#
# RANDOM
#
# jax requires to keep track of the random generator state. This is done by passing a PRNGKey to every random function.
# pcax offers a random generator class that keeps track of the state and can be used to generate random keys. RKG is
# the default random generator used by pcax.
#
########################################################################################################################

# Utils ################################################################################################################


class RKGState(Param):
    """RKGState is a state parameter that tracks a random generator state."""

    def __init__(self, seed: int):
        """RKGState constructor.

        Args:
            seed (int): the initial seed of the random number generator.
        """
        super().__init__(jax.random.PRNGKey(seed))

    def seed(self, seed: int) -> None:
        """Sets a new random seed.

        Args:
            seed (int): the new seed of the random number generator.
        """
        self.set(jax.random.PRNGKey(seed))

    def split(self, n: int) -> jax.typing.ArrayLike:
        """Generates n new keys, updating the internal state.

        Args:
            n (int): the number of keys to generate.
            
        Returns:
            jax.typing.ArrayLike: a list of n keys.
        """
        values = jax.random.split(self.get(), n + 1)
        self.set(values[0])

        return values[1:]


# Random ###############################################################################################################


class RandomKeyGenerator(BaseModule):
    """Random number generator module. Provide an stateful interface to generate random keys accessible in the global
    scope."""

    def __init__(self, seed: int = 0):
        """RandomKeyGenerator constructor.

        Args:
            seed (int, optional): initial seed. Defaults to 0.
        """
        super().__init__()
        self.key = RKGState(seed)

    def seed(self, seed: int = 0):
        """Set a new seed.

        Args:
            seed (int, optional): new seed. Defaults to 0.
        """
        self.key.seed(seed)

    def __call__(self, n: int = 1) -> Tuple[ArrayLike, ...] | ArrayLike:
        """Generate n random keys.
        
        Args:
            n (int, optional): number of keys to generate.
        
        Returns:
            Tuple[ArrayLike, ...] | ArrayLike: a single key if n is 1, otherwise a tuple of keys.
        """
        _k = self.key.split(n)
        
        # For comodity, return a single key if n is 1
        if n == 1:
            return _k[0]
        else:
            return _k


"""Default random generator, globally accessible
Initialize the random generator with a seed based on the current time,
if the user wants to use a different seed, they can call RKG.seed(seed)
"""
RKG = RandomKeyGenerator(time.time_ns())
