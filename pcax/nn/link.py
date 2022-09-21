from ..core.link import Link
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional


class Linear(Link):
    nn: eqx.nn.Linear

    def __init__(
        self,
        _in_features: int,
        _out_features: int,
        _use_bias: bool = True,
        *,
        _key: jax.random.PRNGKey
    ) -> None:
        super().__init__()

        self.nn = eqx.nn.Linear(_in_features, _out_features, _use_bias, key=_key)

    def forward(self, _x: jnp.ndarray, *, _key: Optional[jax.random.PRNGKey] = None):
        return self.nn(_x, key=_key)
