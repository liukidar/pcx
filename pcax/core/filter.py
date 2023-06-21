__all__ = ["_"]

import types
from typing import Any, Tuple, Union
from .parameters import ParameterCache


########################################################################################################################
#
# FILTER
#
# _ is an automatic filter utility to select Parameters by type and attributes within a ParamsDict.
#
########################################################################################################################


class _:
    """Filter class, can be passsed to ParamsDict.filter() to select only the parameters that match the given type
    constraints. By using bitwise logical operators it is possible to specify any type combination to be matched.

    Examples:
        -   _(a|b) & _(c)  # matches all parameters p for which isinstance(p, c) is True and either isinstance(p, a) or
            isinstance(p, b) is True (shortcut for `(_(a) | _(b)) & _(c)`, only possible using the '|' python union type
            operator);
        -   _(a) & ~_(b|c)  # matches all parameters p for which isinstance(p, a) is True and both isinstance(p, b) and
            isinstance(p, b) are False;
        -   `with_cache` specifies whether ParameterCache should match if its referenced parameter would match instead:
            _(a, with_cache=True)  # matches all parameters p s.t. isinstance(p, a) is True or isinstance(p.ref, a) is
            True (and p is of type ParameterCache).
            This is mainly helpful when using pcax.Vectorize as cached activations of a vmapped parameter are to be
            vectorized as well;
        -   _(...)(attr=value)  # matches all parameters matched by _ that have the attribute `attr` equal to `value`
    """

    def __init__(self, arg: Union[type, types.UnionType, '_'], with_cache: bool = False):
        """Filter constructor.

        Args:
            arg: the type, type union or filter on which recursively apply the newly constructed filter.
            with_cache: whether to includes ParameterCache which `ref` would match the filter.
        """

        self.arg = arg
        self.with_cache = with_cache

    def apply(self, var: Any) -> bool:
        """Filters `var`, returning whether it has to be kept (True) or discarded (False) """
        if isinstance(self.arg, _):
            r = self.arg.apply(var)
        elif isinstance(var, ParameterCache) and self.with_cache:
            r = isinstance(var, self.arg) or isinstance(var.ref, self.arg)
        else:
            r = isinstance(var, self.arg)

        return r

    def __or__(self, __other):
        return _or(self, __other)

    def __and__(self, __other):
        return _and(self, __other)

    def __invert__(self):
        return _not(self)

    def __call__(self, **kwargs):
        return _hasattr(self, **kwargs)


class _or:
    """Computes the logical or between two filters or types"""

    def __init__(self, *args: Tuple[_, ...]):
        self.args = args

    def apply(self, var: Any):
        for f in self.args:
            if isinstance(f, _):
                r = f.apply(var)
            else:
                r = isinstance(var, f)

            if r is True:
                return True

        return False


class _and(_):
    """Computes the logical and between two filters or types"""

    def __init__(self, *args: Tuple[_, ...]):
        self.args = args

    def apply(self, var: Any):
        for f in self.args:
            if isinstance(f, _):
                r = f.apply(var)
            else:
                r = isinstance(var, f)

            if r is False:
                return False

        return True


class _not(_):
    """Computes the logical not of a filter or type"""

    def __init__(self, arg: _):
        self.args = arg

    def apply(self, var: Any):
        if isinstance(self.args, _):
            r = self.args.apply(var)
        else:
            r = isinstance(var, self.args)

        return not r


class _hasattr(_):
    """Filters parameters based on their attributes"""

    def __init__(self, arg: _, **attrs):
        self.args = arg
        self.attrs = attrs

    def apply(self, var: Any):
        return self.args.apply(var) and all(
            hasattr(var, attr) and getattr(var, attr) == value
            for attr, value in self.attrs.items()
        )
