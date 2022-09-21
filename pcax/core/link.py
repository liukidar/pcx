from typing import Any

from pcax.core.node import NODE_TYPE, NodeModule


class Link(NodeModule):
    def __init__(self) -> None:
        super().__init__(type=NODE_TYPE.W)

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError("You cannot instantiate an abstract class.")
