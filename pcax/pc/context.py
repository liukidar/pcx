__all__ = ["init_nodes", "init_cache", "vectorize", "gradvalues", "jit"]

from ..core import _, Vectorize, GradValues, Jit
from .variables import NodeVar, ParameterCache
from ..core.parameters import ParamsDict

import contextlib
from typing import Callable, Tuple, Optional, Union


@contextlib.contextmanager
def init_nodes(
    model,
    *args,
    filter=_(NodeVar, ParameterCache),
    in_axis=None,
    out_axis=None,
    clear_on_enter=True,
    **kwargs
):
    if len(args):
        if in_axis is None:
            in_axis = (0,) * len(args)
        if out_axis is None:
            out_axis = (0,)

        def call(*args, model):
            return model(*args)
        r = Vectorize(call, filter, in_axis, out_axis)(*args, model=model, **kwargs)

        if clear_on_enter:
            model.clear_cache()

        yield r
    else:
        yield

    model.clear_cache()
    model.clear_nodes()


@contextlib.contextmanager
def init_cache(model, clear_on_exit=False):
    if clear_on_exit is False:
        model.clear_cache()
    yield

    if clear_on_exit:
        model.clear_cache()


def vectorize(
    filter: Union[_, Callable[[ParamsDict], ParamsDict]],
    in_axis: Tuple[Optional[int], ...] = (0,),
    out_axis: Tuple[Optional[int], ...] = (0,),
    axis_name: str = "batch"
):
    def decorator(f):
        return Vectorize(f, filter, in_axis, out_axis, axis_name)

    return decorator


def gradvalues(
    filter: Union[_, Callable[[ParamsDict], ParamsDict]],
    input_argnums: Optional[Tuple[int, ...]] = None,
):
    def decorator(f):
        return GradValues(f, filter, input_argnums)

    return decorator


def jit(
    filter: Union[_, Callable[[ParamsDict], ParamsDict]] = lambda key, value: True,
    donate_argnums: Tuple[int, ...] = (),
    inline: bool = False
):
    def decorator(f):
        return Jit(f, filter, donate_argnums, inline)

    return decorator
