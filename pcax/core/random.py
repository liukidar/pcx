__all__ = ["RKGState", "RKG", "RandomKeyGenerator"]

from .modules import Module
from .parameters import _BaseParameter, reduce_none

from typing import Optional, List
import time
import jax


class RKGState(_BaseParameter):
    """RKGState are variables that track the random generator state. They are meant to be used internally.
    Currently only the random.RandomKeyGenerator module uses them."""

    def __init__(self, seed: int):
        """RKGState constructor.

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
        """Create multiple seeds from the current seed. This is used internally by Parallel and Vectorize to ensure
        that random numbers are different in parallel threads.

        Args:
            n: the number of seeds to generate.
        """
        values = jax.random.split(self.value, n + 1)
        self._value = values[0]
        return values[1:]


class RandomKeyGenerator(Module):
    """Random number generator module."""

    def __init__(self, seed: int = 0):
        """Create a random key generator, seed is the random generator initial seed."""
        super().__init__()
        self._key: Optional[RKGState] = RKGState(seed)

    @property
    def key(self):
        """The random generator state (a tensor of 2 int32)."""
        return self._key

    def seed(self, seed: int = 0):
        """Sets a new random generator seed."""
        self._key.seed(seed)

    def __call__(self):
        """Generate a new generator state."""
        return self.key.split(1)[0]


RKG = RandomKeyGenerator(time.time_ns())
