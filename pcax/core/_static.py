__all__ = [
    "StaticParam",
    "static"
]


from typing import Any

from ._parameter import BaseParam, get


########################################################################################################################
#
# STATIC
#
# Since Modules are Pytrees, they cannot contains values that are not compatible with JAX (i.e., things that are not
# Arraylike). This means that we cannot store static values in Modules, such as strings, integers, or functions. To
# overcome this limitation, we introduce the StaticParam class, which is a parameter that can store a static value.
# Such static value is ignored by pcax and can be used to store any kind of value.
# For example:
#
# ```python
# class Module(pcax.Module):
#    def __init__(self, x: jax.Array, f: Callable):
#       self.x = x  # NOTE: if we want automatic parameter tracking we should use pcax.Param(x)
#       self.f = f  # NOTE: this is wrong and will raise an error when passing the module to a jax/pcax transformation
#
#   def __call__(y: jax.Array) -> jax.Array:
#       return self.f(x, y)
# ```
#
# is wrong and should, in general, be as follows:

# ```python
# class Module(pcax.Module):
#    def __init__(self, x: jax.Array, f: Callable):
#       self.x = pcax.Param(x)
#       self.f = pcax.static(f)
#
#   def __call__(y: jax.Array) -> jax.Array:
#       return self.f(x, y)  # NOTE: StaticParam overload many methods of the underlying value so that it can be used
#                            # as if it were the value itself.
# ```
########################################################################################################################

# Core #################################################################################################################


class StaticParam(BaseParam):
    """Static parameter class. It is a simple wrapper around a value that makes it static and compatible with JAX/pcax
    transformations. It is a subclass of BaseParam and thus can be used as a parameter in any pcax module.
    NOTE: pcax only tracks dynamic values of parameters, so any change to a static value within a transformation must
    be explicitly handled by the user by returning the new value and updating the original one. However, given the
    nature of jit compilation, it is not advisable to permanently change the value of a static parameter within a
    transformation (as a jitted transformation expects the same static and dynamic input structure every time is
    called). Thus, the default behaviour of pcax is that each change to a static parameter is temporary and does not
    affect the original value outside of a transformation (this may change in the future).

    DEV NOTE: the change can be done by returning parameters instead of value in the transformations, and then update
    the whole original parameter __dict__ instead of only the value. This would allow to keep track of changes to
    static parameters as well (also to update the relevant code that as of now deals only with dynamic parameters).
    """

    def __init__(self, value: Any | None = None):
        """StaticParam constructor.

        Args:
            value (Any | None, optional): static value to be wrapped.
        """
        super().__init__(None)
        self._static_value = value

    def get(self) -> Any:
        """Overloads BaseParam.get to return the static value instead of the dynamic one (i.e., _value, which is None).

        Returns:
            Any: the wrapped static value.
        """
        return self._static_value

    def set(self, value: Any) -> None:
        """Overloads BaseParam.set to set the static value instead of the dynamic one.

        Args:
            value (Any): the new value to be wrapped.
        """
        self._static_value = value

    # Overload of common methods to directly access the static value.

    def __getattr__(self, __name: str) -> Any:
        """Overloads __getattr__ to return the attribute of the static value."""
        return getattr(self._static_value, __name)

    def __contains__(self, __key: str) -> bool:
        """Overloads __contains__ to check if the static value contains a key."""
        return __key in self._static_value

    def __iter__(self) -> Any:
        """Overloads __iter__ to iterate over the static value."""
        return iter(self._static_value)

    def __len__(self) -> int:
        """Overloads __len__ to return the length of the static value."""
        return len(self._static_value)

    def __getitem__(self, __idx: Any) -> Any:
        """Overloads __getitem__ to get the item at index __idx in the static value."""
        return self._static_value.__getitem__(__idx)

    def __setitem__(self, __idx: Any, value: Any) -> Any:
        """Overloads __setitem__ to set the item at index __idx in the static value."""
        return self._static_value.__setitem__(__idx, Any)

    def __call__(self, *args, **kwds) -> Any:
        """Overloads __call__ to call the static value as if it were a function."""
        return self._static_value(*args, **kwds)

    def __eq__(self, __value: object) -> bool:
        # 'get' handles '__value' being a Param or not.
        return self.get() == get(__value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._static_value)})"


# Utils ################################################################################################################


def static(x: Any | StaticParam) -> StaticParam:
    """Wraps a value into a StaticParam, making it static and thus compatible with JAX transformations.

    Args:
        x (Any | StaticParam): value to be wrapped. If x is already a StaticParam, it is returned as is.

    Returns:
        StaticParam: the static parameter wrapping the input value.
    """
    return x if isinstance(x, StaticParam) else StaticParam(x)
