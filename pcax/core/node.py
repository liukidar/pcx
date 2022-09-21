import equinox as eqx
from enum import IntEnum, auto


class NodeAttr(str):
    def __new__(cls, *args, **kw):
        return str.__new__(cls, *args, **kw)


class NODE_TYPE(IntEnum):
    NONE = auto()
    W = auto()
    X = auto()


class NODE_STATUS(IntEnum):
    NONE = auto()
    FROZEN = auto()


class NodeInfo:
    type: NODE_TYPE
    status: NODE_STATUS

    def __init__(self, **kwargs) -> None:
        self.type = kwargs.get("type", NODE_TYPE.NONE)
        self.status = kwargs.get("status", NODE_STATUS.NONE)


class NodeModule(eqx.Module):
    _node_info: NodeInfo = eqx.static_field()

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._node_info = NodeInfo(**kwargs)
