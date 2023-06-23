__all__ = ["train", "eval", "init_cache", "vectorize", "gradvalues", "jit"]

from ..core import f, Vectorize, GradValues, Jit, Module
from .variables import NodeVar
from ..core.parameters import ParamsDict
from .module import EnergyModule

import contextlib
from typing import Callable, Tuple, Optional, Union


@contextlib.contextmanager
def train(
    model: Union[Module, EnergyModule],
    *args,
    filter: Union[f, Callable] = f(NodeVar, with_cache=True),
    in_axis: Optional[Tuple[Union[None, int]]] = None,
    out_axis: Optional[Tuple[Union[None, int, str]]] = None,
    **kwargs
):
    model.train()

    if len(args):
        yield init_module(model, *args, filter=filter, in_axis=in_axis, out_axis=out_axis, **kwargs)
    else:
        yield

    model.clear_cache()


@contextlib.contextmanager
def eval(
    model: Union[Module, EnergyModule],
    *args,
    filter: Union[f, Callable] = f(NodeVar, with_cache=True),
    in_axis: Optional[Tuple[Union[None, int]]] = None,
    out_axis: Optional[Tuple[Union[None, int, str]]] = None,
    **kwargs
):
    model.eval()

    if len(args):
        yield init_module(model, *args, filter=filter, in_axis=in_axis, out_axis=out_axis, **kwargs)
    else:
        yield

    model.clear_cache()


def init_module(
    model: Module | EnergyModule,
    *args,
    filter: Union[f, Callable] = f(NodeVar, with_cache=True),
    in_axis: Optional[Tuple[Union[None, int]]] = None,
    out_axis: Optional[Tuple[Union[None, int, str]]] = None,
    **kwargs
):
    model.set_status(init=True)

    in_axis = in_axis or ((0,) * len(args))
    out_axis = out_axis or (0,)

    @vectorize(filter=filter, in_axis=in_axis, out_axis=out_axis)
    def forward(*args, model):
        return model(*args)

    r = forward(*args, model=model, **kwargs)

    model.set_status(init=False)

    return r


@contextlib.contextmanager
def init_cache(model, clear_on_exit=False):
    if clear_on_exit is False:
        model.clear_cache()
    yield

    if clear_on_exit:
        model.clear_cache()


def vectorize(
    filter: Union[f, Callable[[ParamsDict], ParamsDict]],
    in_axis: Tuple[Optional[int], ...] = (0,),
    out_axis: Tuple[Optional[int], ...] = (0,),
    axis_name: str = "batch"
):
    def decorator(fn):
        return Vectorize(fn, filter, in_axis, out_axis, axis_name)

    return decorator


def gradvalues(
    filter: Union[f, Callable[[ParamsDict], ParamsDict]],
    input_argnums: Tuple[int, ...] = (),
):
    def decorator(fn):
        return GradValues(fn, filter, input_argnums)

    return decorator


def jit(
    filter: Union[f, Callable[[ParamsDict], ParamsDict]] = lambda key, value: True,
    donate_argnums: Tuple[int, ...] = (),
    inline: bool = False
):
    def decorator(fn):
        return Jit(fn, filter, donate_argnums, inline)

    return decorator
