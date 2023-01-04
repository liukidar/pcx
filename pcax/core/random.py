__all__ = ["DEFAULT_GENERATOR", "Generator"]

from .structure import RandomState, VarCollection, Module

from typing import Optional
import time


class Generator(Module):
    """Random number generator module."""

    def __init__(self, seed: int = 0):
        """Create a random key generator, seed is the random generator initial seed."""
        super().__init__()
        self.initial_seed = seed
        self._key: Optional[RandomState] = None

    @property
    def key(self):
        """The random generator state (a tensor of 2 int32)."""
        if self._key is None:
            self._key = RandomState(self.initial_seed)
        return self._key

    def seed(self, seed: int = 0):
        """Sets a new random generator seed."""
        self.initial_seed = seed
        if self._key is not None:
            self._key.seed(seed)

    def __call__(self):
        """Generate a new generator state."""
        return self.key.split(1)[0]

    def vars(self, scope: str = "") -> VarCollection:
        self.key  # Make sure the key is created before collecting the vars.
        return super().vars(scope=scope)

    def __repr__(self):
        return f"{self.__class__.__name__}(seed={self.initial_seed})"


DEFAULT_GENERATOR = Generator(time.time_ns())
