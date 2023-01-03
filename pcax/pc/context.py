__all__ = ["init_nodes", "init_cache", "bind", "vectorize", "gradvalues", "jit"]

from ..core import Function, VarCollection, Vectorize, GradValues, Jit, _
from ..core.util import make_args
from .variables import NodeVar

import functools
import contextlib


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

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                _original = {k: f.__globals__.get(k, None) for k in kwarg_vars.keys()}
                f.__globals__.update(kwarg_vars)
                return f(*args, **kwargs)
            finally:
                f.__globals__.update(_original)

        return Function(wrapper, vc)

    return decorator


@contextlib.contextmanager
def init_nodes(model, *args, filter=_(NodeVar), in_axis=None, out_axis=None, **kwargs):
    if len(args):
        if in_axis is None:
            in_axis = (0,) * len(args)
        if out_axis is None:
            out_axis = (0,)
        yield Vectorize(bind(model)(model.__call__), filter, in_axis, out_axis)(*args, **kwargs)
    else:
        yield

    model.clear_cache()
    model.clear_nodes()


@contextlib.contextmanager
def init_cache(model, clear_on_exit=False):
    model.clear_cache()
    yield

    if clear_on_exit:
        model.clear_cache()


def vectorize(*args, **kwargs):
    @functools.wraps(Vectorize)
    def decorator(f):
        return Vectorize(f, *args, **kwargs)

    return decorator


def gradvalues(*args, **kwargs):
    @functools.wraps(GradValues)
    def decorator(f):
        return GradValues(f, *args, **kwargs)

    return decorator


def jit(*args, **kwargs):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(static_kwargs, *args, **kwargs):
            # Finds the original function wrapped in the f hierarchy of transformations
            # and calls it by merging all the given arguments.
            return f(*make_args(Function.leaf_fn(f), args, {**dict(static_kwargs), **kwargs}))

        jit_f = Jit(Function(wrapper, f.vc), *args, **kwargs, static_argnums=(0,))

        return lambda **static_kwargs: functools.partial(jit_f, tuple(static_kwargs.items()))

    return decorator
