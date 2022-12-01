__all__ = ["eval", "train", "bind", "vectorize", "gradvalues", "jit"]

from ..core import Function, VarCollection, Vectorize, GradValues, Jit

import functools
import contextlib


@contextlib.contextmanager
def eval(module):
    yield
    module.clear(True)


@contextlib.contextmanager
def train(module):
    yield
    module.clear(False)


def bind(*arg_vars, **kwarg_vars):
    """Decorator to bind a function to a set of variables."""

    def decorator(f):
        vc = functools.reduce(
            lambda x, y: x + y, (m.vars() for m in arg_vars), VarCollection()
        ) + functools.reduce(
            lambda x, y: x + y,
            (m.vars().rename(k) for k, m in kwarg_vars.items()),
            VarCollection(),
        )

        def wrapper(*args, **kwargs):
            try:
                _original = {k: f.__globals__.get(k, None) for k in kwarg_vars.keys()}
                f.__globals__.update(kwarg_vars)
                return f(*args, **kwargs)
            finally:
                f.__globals__.update(_original)

        return Function(wrapper, vc)

    return decorator


def vectorize(*args, **kwargs):
    def decorator(f):
        return Vectorize(f, *args, **kwargs)

    return decorator


def gradvalues(*args, **kwargs):
    def decorator(f):
        return GradValues(f, *args, **kwargs)

    return decorator


def jit(*args, **kwargs):
    def decorator(f):
        return Jit(f, *args, **kwargs)

    return decorator
