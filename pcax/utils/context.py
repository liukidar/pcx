__all__ = ["train", "eval", "step", "vectorize", "grad_and_values", "jit"]

import contextlib
from typing import Callable, Optional, Tuple, Union

from ..core import GradAndValues, Jit, Module, Vectorize, f
from ..core.parameters import ParamDict
from ..pc.energymodule import EnergyModule
from ..pc.parameters import NodeParam, LayerParam
from .optim import Optim


@contextlib.contextmanager
def train(
    model: Union[Module, EnergyModule],
    *args,
    filter: Union[f, Callable] = f(NodeParam, with_cache=True),
    in_axis: Optional[Tuple[Union[None, int]]] = None,
    out_axis: Optional[Tuple[Union[None, int, str]]] = None,
    **kwargs,
):
    model.train()
    if isinstance(model, EnergyModule):
        model.clear_cache()

    if len(args):
        yield init_module(
            model, *args, filter=filter, in_axis=in_axis, out_axis=out_axis, **kwargs
        )
    else:
        yield

    if isinstance(model, EnergyModule):
        model.clear_cache()


@contextlib.contextmanager
def warmup_for_train(
    model: EnergyModule,
    optim_x: Optim,
    loss_fn: Callable,
    *loss_args,
    loss_param_filter: Union[f, Callable[[ParamDict], ParamDict]] = (
        f(NodeParam)(frozen=False) | f(LayerParam)
    ),
):
    _grad_and_values = grad_and_values(loss_param_filter)(loss_fn)

    model.train()
    model.clear_cache()
    model.set_status(init=True)
    try:
        gradients, (energy,) = _grad_and_values(*loss_args, model=model)
        model.set_status(init=False)
        optim_x(gradients)
        yield energy
    finally:
        model.set_status(init=False)
        model.clear_cache()


@contextlib.contextmanager
def eval(
    model: Union[Module, EnergyModule],
    *args,
    filter: Union[f, Callable] = f(NodeParam, with_cache=True),
    in_axis: Optional[Tuple[Union[None, int]]] = None,
    out_axis: Optional[Tuple[Union[None, int, str]]] = None,
    **kwargs,
):
    model.eval()
    if isinstance(model, EnergyModule):
        model.clear_cache()

    if len(args):
        yield init_module(
            model, *args, filter=filter, in_axis=in_axis, out_axis=out_axis, **kwargs
        )
    else:
        yield

    if isinstance(model, EnergyModule):
        model.clear_cache()


def init_module(
    model: Module | EnergyModule,
    *args,
    filter: Union[f, Callable] = f(NodeParam, with_cache=True),
    in_axis: Optional[Tuple[Union[None, int]]] = None,
    out_axis: Optional[Tuple[Union[None, int, str]]] = None,
    **kwargs,
):
    if isinstance(model, EnergyModule):
        model.set_status(init=True)

    in_axis = in_axis or ((0,) * len(args))
    out_axis = out_axis or (0,)

    @vectorize(filter=filter, in_axis=in_axis, out_axis=out_axis)
    def forward(*args, model, **kwargs):
        return model(*args, **kwargs)

    r = forward(*args, model=model, **kwargs)

    if isinstance(model, EnergyModule):
        model.set_status(init=False)

    return r


@contextlib.contextmanager
def step(model):
    if isinstance(model, EnergyModule):
        model.clear_cache()

    yield

    if isinstance(model, EnergyModule):
        model.clear_cache()


# TRANSFORMS #########################################################################################################


def vectorize(
    filter: Union[f, Callable[[ParamDict], ParamDict]] = lambda key, value: False,
    in_axis: Tuple[Optional[int], ...] = (0,),
    out_axis: Tuple[Optional[int], ...] = (0,),
    axis_name: str = "batch",
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
    inline: bool = False,
):
    def decorator(fn):
        return Jit(fn, filter, donate_argnums, inline)

    return decorator
