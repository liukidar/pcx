__all__ = ["train", "eval", "step", "vectorize", "grad_and_values", "jit"]

from ..core import f, Vectorize, GradAndValues, Jit, Module
from .parameters import NodeParam
from ..core.parameters import ParamDict
from .energymodule import EnergyModule

import contextlib
from typing import Callable, Tuple, Optional, Union


@contextlib.contextmanager
def train(
    model: Union[Module, EnergyModule],
    *args,
    filter: Union[f, Callable] = f(NodeParam, with_cache=True),
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
    filter: Union[f, Callable] = f(NodeParam, with_cache=True),
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
    filter: Union[f, Callable] = f(NodeParam, with_cache=True),
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
def step(model):
    model.clear_cache()

    yield

    model.clear_cache()


# TRANSFORMS #########################################################################################################


def vectorize(
    filter: Union[f, Callable[[ParamDict], ParamDict]],
    in_axis: Tuple[Optional[int], ...] = (0,),
    out_axis: Tuple[Optional[int], ...] = (0,),
    axis_name: str = "batch"
):
    def decorator(fn):
        return Vectorize(fn, filter, in_axis, out_axis, axis_name)

    return decorator


def grad_and_values(
    filter: Union[f, Callable[[ParamDict], ParamDict]],
    input_argnums: Tuple[int, ...] = (),
):
    def decorator(fn):
        return GradAndValues(fn, filter, input_argnums)

    return decorator


def jit(
    filter: Union[f, Callable[[ParamDict], ParamDict]] = lambda key, value: True,
    donate_argnums: Tuple[int, ...] = (),
    inline: bool = False
):
    def decorator(fn):
        return Jit(fn, filter, donate_argnums, inline)

    return decorator
