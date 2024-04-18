__all__ = ["Mask", "m"]

from typing import Any, Tuple, Dict, Type, Callable
import types

import jax.tree_util as jtu

from ..core._parameter import Param, BaseParam
from ..core._tree import tree_ref


########################################################################################################################
#
# MASK
#
# Mask provided a convenient way to filter parameters based on their type or attributes. It is used to mask out 
# parameters that are not relevant to the current operation or transformation (such as when we want to compute the
# gradient only with respect to a subset of the parameters). It is quite similar to `equinox.filter` as functionality
# but it provides shortcuts to filter based on the type of the parameter or its attributes. Further classes can be
# defined to provide more complex filtering operations.
#
########################################################################################################################


class Mask:
    """Mask can either be used to remove unwanted tensors (by setting them to None, similarly to `equinox.filter`) or
    to map them to a different value based on the whether they pass the filter or not (by using `map_to`). It is
    important to note that, in any case, the filter acts on the Params of a pytree, seeing them as leaves, while the
    modified value is the Param's value (and thus it is impossible to differentiate between different children of the
    same Param, in the case its value is a pytree itself).
    """
    def __init__(
        self,
        x: Type | 'm' | Callable,
        map_to: Tuple[Any, Any] | None = None
    ):
        """Mask constructor.

        Args:
            x (Type | m | Callable): The object on which recursively apply the mask. If it is a type, it will
                be used to filter the parameters by type. If it is a callable, it must return a boolean and it will be
                used to filter the parameters by the result of the call. If it is a mask (i.e., `m`), it will be
                recursively applied to the parameters.
            map_to (Tuple[Any, Any] | None, optional): if not None, it will be used to set the parameters to either
                value based on the mask boolean result. If None, the parameters will be set to None if mask is False,
                and will remain unchanged otherwise. Defaults to None.
        """
        self.x = x
        self.map_to = map_to

    def __call__(self, pydag: Any, is_pytree: bool = False) -> Any:
        """Applies the mask to the given pydag.

        Args:
            pydag (Any): target pydag.
            is_pytree (bool, optional): to guarantee that each parameter is masked only once (and thus the mask
                will not have duplicates), we have to ensure we are working with a pytree. Defaults to False.

        Returns:
            Any: the masked pydag (enforced to be a pytree).
        """
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
        """Recursively applies the mask to a leaf.

        Returns:
            Any: the masked leaf.
        """
        if isinstance(mask, type | types.UnionType):
            r = isinstance(leaf, mask)
        elif mask is None:
            r = True
        else:
            r = mask(leaf)

        return r


class m:
    """m is a utility mask that can be used to combine different filters using logical operators (and more).
    It is necessary as python supports only the `|` operator for types.
    Available operators are:
    - `|` for logical or (note that we can also use the `|` operator directly on the type themselves):
        `float | int` is equivalent to `m(int) | m(float)`. In this case, the parameter will be masked
        if it is of either type. The other operators behave similarly.
    - `&` for logical and
    - `~` for logical not
    - `has` to filter based on the presence of an attribute with a specific value
    - `has_not` to filter based on the absence of an attribute with a specific value
    
    For example:
    
    ```python
    (m(A | B)) & ~m(C).has(attr1=1)
    ```
    
    selects paraameters of class A or B that are not of class C and have an attribute `attr1` equal to 1. 
    """

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
    """Computes the logical or between two masks or types"""

    def __init__(self, *args: Tuple[Any, ...]):
        super().__init__(args)

    def __call__(self, leaf: Any):
        return any(Mask.apply(i, leaf) for i in self.x)


class _m_and(Mask):
    """Computes the logical and between two masks or types"""

    def __init__(self, *args: Tuple[Any, ...]):
        super().__init__(args)

    def __call__(self, leaf: Any):
        return all(Mask.apply(i, leaf) for i in self.x)


class _m_not(Mask):
    """Computes the logical not of a mask or type"""

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
