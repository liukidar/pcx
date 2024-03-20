__all__ = ["f"]

from typing import Any, Tuple, Dict, Type, Callable
import types

import jax.tree_util as jtu

from ..core._parameter import Param, BaseParam
from ..core._tree import tree_ref


########################################################################################################################
#
# FILTER
#
# f is an automatic filter utility to select Parameters by type and attributes within a ParamsDict.
#
########################################################################################################################


class Mask:
    def __init__(
        self,
        x: Type | 'm' | Callable | Dict,
        map_to: Tuple[Any, Any] | None = None
    ):
        self.x = x
        self.map_to = map_to
    
    def __call__(self, pydag: Any, is_pytree: bool = False) -> Any:
        if self.map_to is None:
            def map_fn(param: Any | Param):
                _mask_value = Mask.apply(self.x, param)
                
                return param if _mask_value is True else None
        else:
            def map_fn(param: Any | Param):
                return self.map_to[Mask.apply(self.x, param)]
            
        t = jtu.tree_map(
            map_fn,
            pydag if is_pytree else tree_ref(pydag),
            is_leaf=lambda x: isinstance(x, BaseParam)
        )
        
        return t
        
    @staticmethod
    def apply(mask, leaf: Any) -> Any:
        if isinstance(mask, type | types.UnionType):
            r = isinstance(leaf, mask)
        elif mask is None:
            r = True
        else:
            r = mask(leaf)

        return r


class m:
    def __init__(
        self,
        x: Type | 'm' | Callable | None = None,
    ):
        self.x = x
    
    def __call__(self, leaf: Any) -> Any:
        return Mask.apply(self.x, leaf)

    def __or__(self, __other):
        return _m_or(self, __other)

    def __and__(self, __other):
        return _m_and(self, __other)

    def __invert__(self):
        return _m_not(self)

    def has(self, **kwargs):
        return _m_hasattr(self, **kwargs)

    def has_not(self, **kwargs):
        return _m_hasnotattr(self, **kwargs)


class _m_or(Mask):
    """Computes the logical or between two filters or types"""

    def __init__(self, *args: Tuple[Any, ...]):
        super().__init__(args)

    def __call__(self, leaf: Any):
        return any(Mask.apply(i, leaf) for i in self.x)


class _m_and(Mask):
    """Computes the logical and between two filters or types"""

    def __init__(self, *args: Tuple[Any, ...]):
        super().__init__(args)

    def __call__(self, leaf: Any):
        return all(Mask.apply(i, leaf) for i in self.x)


class _m_not(Mask):
    """Computes the logical not of a filter or type"""

    def __init__(self, arg: Any):
        super().__init__(arg)

    def __call__(self, leaf: Any):
        return not Mask.apply(self.x, leaf)


class _m_hasattr(m):
    """Filters parameters based on their attributes"""

    def __init__(self, arg: m, **attrs: Dict[str, Any]):
        super().__init__(arg)
        self.attrs = attrs

    def __call__(self, leaf: Any):
        return Mask.apply(self.x, leaf) and all(
            hasattr(leaf, attr) and getattr(leaf, attr) == value
            for attr, value in self.attrs.items()
        )


class _m_hasnotattr(m):
    """Filters parameters based on their attributes"""

    def __init__(self, arg: m, **attrs: Dict[str, Any]):
        super().__init__(arg)
        self.attrs = attrs

    def __call__(self, leaf: Any):
        return Mask.apply(self.x, leaf) and all(
            (not hasattr(leaf, attr)) or (getattr(leaf, attr) != value)
            for attr, value in self.attrs.items()
        )
