import jax.numpy as jnp
from typing import Any

from pcax.core.nn import NodeModule

class Link(NodeModule):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)

	def __call__(self, *_args, **_kwargs) -> Any:
		return self.forward(*_args, **_kwargs)

	def forward(self, _x):
		raise NotImplementedError("You cannot instantiate an abstract class.")