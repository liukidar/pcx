__all__ = ["M", "M_is", "M_has", "M_hasnot"]

from typing import Any, Tuple, Dict, Type, Callable
import types

import jax.tree_util as jtu

from ..core._parameter import Param, BaseParam
from ..core._tree import tree_ref


#######################################################################################################################
#
# MASK
#
# Mask provided a convenient way to filter parameters based on their type or attributes. It is used to mask out
# parameters that are not relevant to the current operation or transformation (such as when we want to compute the
# gradient only with respect to a subset of the parameters). It is quite similar to `equinox.filter` as functionality
# but it provides shortcuts to filter based on the type of the parameter or its attributes. Further classes can be
# defined to provide more complex filtering operations.
#
#######################################################################################################################


class M:
    """M applies a mask can either be used to remove unwanted tensors (by setting them to None, similarly to
    `equinox.filter`) or to map them to a different value based on the whether they pass the filter or not (by setting
    `map_to` via the `.to` function). It is important to note that, in any case, the filter acts on the Params of a
    pytree, seeing them as leaves, while the modified value is the Param's value (and thus it is impossible to
    differentiate between multiple children of the same Param, in the case its value is a pytree itself).

    M that can be used to combine different filters using logical operators (and more).
    It is necessary as python supports only the `|` operator for types.
    Available operators are:
    - `|` for logical or (note that we can also use the `|` operator directly on the type themselves):
        `float | int` is equivalent to `M(int) | M(float)`. In this case, the parameter will be masked
        if it is of either type. The other operators behave similarly.
    - `&` for logical and
    - `~` for logical not
    - `has` to filter based on the presence of an attribute with a specific value
    - `hasnot` to filter based on the absence of an attribute with a specific value

    For example:

    ```python
    (M(A | B)) & ~M_has(C, attr1=1)
    ```

    selects paraameters of class A or B that are not of class C and have an attribute `attr1` equal to 1.
    """

    def __init__(self, mask: Type | types.UnionType | "M" | Callable):
        """Mask constructor.

        Args:
            m (Type | types.UnionType | "mask" | Callable): The object on which recursively apply the mask.
                If it is a type, it will be used to filter the parameters by type. If it is a callable, it must return
                a boolean and it will be used to filter the parameters by the result of the call. If it is a mask,
                it will be recursively applied to the parameters.
        """
        self.mask = mask
        self.map_to = None

    def __call__(
        self, pydag: Any, is_pytree: bool = False
    ) -> Any:
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
                _mask_value = self.apply(param)

                return param if _mask_value is True else None
        else:

            def map_fn(param: Any | Param):
                return self.map_to[self.apply(param)]

        t = jtu.tree_map(
            map_fn,
            pydag if is_pytree else tree_ref(pydag),
            is_leaf=lambda x: isinstance(x, BaseParam),
        )

        return t
    
    def to(self, map_to: Tuple[Any, Any] | None) -> "M":
        """Sets the `map_to` value, this allows to map the target pytree to the desired pair of masked/unmasked values,
        instead of relying on the default behaviour of simply filtering out (i.e., substite with `None`) values that
        do not meet the mask requirements.

        Args:
            map_to (Tuple[Any, Any] | None): if not None, it will be used to set the parameters to either
                value based on the mask boolean result. If None, the parameters will be set to None if mask is False,
                and will remain unchanged otherwise. Defaults to None.

        Returns:
            M: returns the mask itself to allow for inline epxressions, such as
                `masked_model = pxu.M(pxnn.LayerParam).to([False, True])(model)`.
        """
        self.map_to = map_to
        
        return self

    @staticmethod
    def _resolve(mask: Type | types.UnionType | "M" | Callable, leaf: Any) -> Any:
        """Recursively applies the mask to a leaf.

        Returns:
            Any: the masked leaf.
        """
        if isinstance(mask, type | types.UnionType):
            r = isinstance(leaf, mask)
        elif isinstance(mask, M):
            r = mask.apply(leaf)
        elif mask is None:
            r = True
        else:
            r = mask(leaf)

        return r

    def apply(self, leaf: Any) -> Any:
        return M._resolve(self.mask, leaf)

    def __invert__(self) -> "_M_not":
        return _M_not(self)

    def __or__(self, __other) -> "_M_or":
        return _M_or(self, __other)

    def __and__(self, __other) -> "_M_and":
        return _M_and(self, __other)

    def has(self, **kwargs) -> "_M_hasattr":
        return _M_hasattr(self, **kwargs)

    def hasnot(self, **kwargs) -> "_M_hasnotattr":
        return _M_hasnotattr(self, **kwargs)


class _M_not(M):
    """Computes the logical not of a mask or type"""

    def __call__(self, leaf: Any):
        return not M._resolve(self.mask, leaf)


class _M_or(M):
    """Computes the logical or between two masks or types"""

    def __init__(self, *mask: Tuple[Any, ...]):
        super().__init__(mask)

    def apply(self, leaf: Any):
        return any(M._resolve(i, leaf) for i in self.mask)


class _M_and(_M_or):
    """Computes the logical and between two masks or types"""

    def apply(self, leaf: Any):
        return all(M._resolve(i, leaf) for i in self.mask)


class _M_hasattr(M):
    """Filters parameters based on their attributes"""

    def __init__(self, mask: M, **attrs: Dict[str, Any]):
        super().__init__(mask)
        self.attrs = attrs

    def apply(self, leaf: Any):
        return M._resolve(self.mask, leaf) and all(
            hasattr(leaf, attr) and getattr(leaf, attr) == value
            for attr, value in self.attrs.items()
        )


class _M_hasnotattr(_M_hasattr):
    """Filters parameters based on their attributes"""

    def apply(self, leaf: Any):
        return M._resolve(self.mask, leaf) and all(
            (not hasattr(leaf, attr)) or (getattr(leaf, attr) != value)
            for attr, value in self.attrs.items()
        )


# Public methods ######################################################################################################


def M_is(*mask: Tuple[Any, ...]) -> _M_and:
    """Creates a mask that filters parameters if they satisfy all the conditions.
    Equivalent to `M(mask[0]) & M(mask[1]) & ... & M(mask[n])`."""

    return _M_and(*mask)


def M_has(mask: M, **attrs: Dict[str, Any]) -> _M_hasattr:
    """Creates a mask that filters parameters if they satisfy all the conditions.
    Equivalent to `M(mask).has(**attrs)`."""

    return _M_hasattr(mask, **attrs)


def M_hasnot(mask: M, **attrs: Dict[str, Any]) -> _M_hasnotattr:
    """Creates a mask that filters parameters if they satisfy all the conditions.
    Equivalent to `M(mask).hasnot(**attrs)`."""

    return _M_hasnotattr(mask, **attrs)
