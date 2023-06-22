__all__ = ["f"]

import types
from typing import Any, Tuple, Union, Dict
from .parameters import ParameterCache


########################################################################################################################
#
# FILTER
#
# f is an automatic filter utility to select Parameters by type and attributes within a ParamsDict.
#
########################################################################################################################


class f:
    """Filter class, can be passsed to ParamsDict.filter() to select only the parameters that match the given type
    constraints. By using bitwise logical operators it is possible to specify any type combination to be matched.

    Examples:
        -   f(a|b) & f(c)  # matches all parameters p for which isinstance(p, c) is True and either isinstance(p, a) or
            isinstance(p, b) is True (shortcut for `(f(a) | f(b)) & f(c)`, only possible using the '|' python union type
            operator);
        -   f(a) & ~f(b|c)  # matches all parameters p for which isinstance(p, a) is True and both isinstance(p, b) and
            isinstance(p, b) are False;
        -   `with_cache` specifies whether ParameterCache should match if its referenced parameter would match instead:
            f(a, with_cache=True)  # matches all parameters p s.t. isinstance(p, a) is True or isinstance(p.ref, a) is
            True (and p is of type ParameterCache).
            This is mainly helpful when using pcax.Vectorize as cached activations of a vmapped parameter are to be
            vectorized as well;
        -   f(...)(attr=value)  # matches all parameters matched by f that have the attribute `attr` equal to `value`
    """

    def __init__(self, arg: Union[type, types.UnionType, 'f'], with_cache: bool = False):
        """Filter constructor.

        Args:
            arg: the type, type union or filter on which recursively apply the newly constructed filter.
            with_cache: whether to includes ParameterCache which `ref` would match the filter.
        """

        self.arg = arg
        self.with_cache = with_cache

    def apply(self, var: Any) -> bool:
        """Filters `var`, returning whether it has to be kept (True) or discarded (False) """
        if isinstance(self.arg, f):
            r = self.arg.apply(var)
        elif isinstance(var, ParameterCache) and self.with_cache:
            r = isinstance(var, self.arg) or isinstance(var.ref, self.arg)
        else:
            r = isinstance(var, self.arg)

        return r

    def __or__(self, __other):
        return _f_or(self, __other)

    def __and__(self, __other):
        return _f_and(self, __other)

    def __invert__(self):
        return _f_not(self)

    def __call__(self, **kwargs):
        return _f_hasattr(self, **kwargs)


class _f_or:
    """Computes the logical or between two filters or types"""

    def __init__(self, *args: Tuple[f, ...]):
        self.args = args

    def apply(self, var: Any):
        for a in self.args:
            if isinstance(a, f):
                r = a.apply(var)
            else:
                r = isinstance(var, f)

            if r is True:
                return True

        return False


class _f_and(f):
    """Computes the logical and between two filters or types"""

    def __init__(self, *args: Tuple[f, ...]):
        self.args = args

    def apply(self, var: Any):
        for a in self.args:
            if isinstance(a, f):
                r = a.apply(var)
            else:
                r = isinstance(var, f)

            if r is False:
                return False

        return True


class _f_not(f):
    """Computes the logical not of a filter or type"""

    def __init__(self, arg: f):
        self.args = arg

    def apply(self, var: Any):
        if isinstance(self.args, f):
            r = self.args.apply(var)
        else:
            r = isinstance(var, self.args)

        return not r


class _f_hasattr(f):
    """Filters parameters based on their attributes"""

    def __init__(self, arg: f, **attrs: Dict[str, Any]):
        self.args = arg
        self.attrs = attrs

    def apply(self, var: Any):
        return self.args.apply(var) and all(
            hasattr(var, attr) and getattr(var, attr) == value
            for attr, value in self.attrs.items()
        )
