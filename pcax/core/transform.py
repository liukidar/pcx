__all__ = [
    "Transform",
    "Jit",
    "Vectorize",
    "GradValues",
]

from typing import Optional, List, Union, Callable, Tuple, Any, Dict
import inspect
import functools
import abc
import copy

import jax
import jax.tree_util as jt

from pcax.core.filter import _

from ..core.modules import Module, Function
from ..core.random import RKG
from ..core.util import repr_function, move, hash_pytree
from ..core.parameters import ParamsDict
from ..core.random import _RKGState


class Transform(abc.ABC):
    def __init__(self, f: Union['Transform', Callable], filter: Union[_, Callable[[ParamsDict], ParamsDict]]):
        if isinstance(f, Transform):
            self.__wrapped__ = f.__wrapped__
            self.f = copy.deepcopy(f)
        else:
            self.__wrapped__ = f
            self.f = f
        self.kwargs = {}
        self.params: Optional[ParamsDict] = None
        self.filter = filter
        self.transform = None

    def __call__(self, *args, **kwargs: Any) -> Any:
        f = self._build(
            functools.reduce(
                lambda x, y: x + y,
                (m.parameters().rename(k) for k, m in kwargs.items() if isinstance(m, Module)),
                RKG.parameters(),
            ),
            kwargs
        )
        return f(*args)

    def _build(self, params, kwargs):
        if isinstance(self.f, Transform):
            self.f = self.f._build(params, kwargs)

        self.params = params
        self.kwargs = kwargs

        if self.transform is None:
            self.transform = self._make_transform()

        def f(*args):
            return self._call(self.partition, *args)

        return Function(f, self.params)

    def _functional(self, t: Callable) -> Callable:
        def wrapper(params_copy, *args, **kwargs):
            params_partition = tuple(move(c, p) for c, p in zip(params_copy, self.partition))
            if isinstance(self.f, Function):
                output = t(*args, **kwargs)
            else:
                output = t(*args, **self.kwargs, **kwargs)

            return output, params_partition

        return wrapper

    @property
    def partition(self) -> Tuple[ParamsDict, ParamsDict]:
        target = self.params.filter(self.filter) + RKG.parameters()

        return target, self.params - target

    @abc.abstractmethod
    def _call(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_transform(self):
        raise NotImplementedError()

    def __repr__(self):
        f = (
            repr(self.__wrapped__)
            if isinstance(self.__wrapped__, Transform)
            else repr_function(self.__wrapped__)
        )
        return f"{self.__class__.__name__}(f={f})"


class Jit(Transform):
    def __init__(
        self,
        f: Union[Transform, Callable],
        filter: Union[_, Callable[[ParamsDict], ParamsDict]] = lambda *args: True,
        donate_argnums: Tuple[int, ...] = (),
        inline: bool = True,
    ):
        super().__init__(f, filter)

        self.donate_argnums = donate_argnums
        self.inline = inline
        self.static_args_hash = None

    def snapshot(self, **kwargs: Module):
        t = copy.deepcopy(self)
        t.transform = None
        t.kwargs = {}

        return t._build(
            functools.reduce(
                lambda x, y: x + y,
                (m.parameters().rename(k) for k, m in kwargs.items() if isinstance(m, Module)),
                RKG.parameters(),
            ),
            kwargs
        )

    def _call(self, params_partition, *args):
        output, params_partition = self.transform(
            params_partition, self.static_args_hash, *args
        )

        return output

    def _build(self, params, kwargs):
        self.static_args_hash = hash_pytree(kwargs)

        return super()._build(params, kwargs)

    def _make_transform(self):
        return jax.jit(
            lambda params, _, *args: self._functional(self.f)(params, *args),
            static_argnums=(1,),
            donate_argnums=(0,) + tuple(x + 2 for x in self.donate_argnums),
            inline=self.inline,
        )


class Vectorize(Transform):
    def __init__(
        self,
        f: Union[Module, Callable],
        filter: Union[_, Callable[[ParamsDict], ParamsDict]],
        in_axis: Tuple[Optional[int], ...] = (0,),
        out_axis: Tuple[Optional[int], ...] = (0,),
        axis_name: str = "batch",
    ):
        super().__init__(f, filter)

        self.in_axis = in_axis
        self.out_axis = out_axis
        self.axis_name = axis_name

    def _call(self, params_partition, *args):
        params_copy = tuple(move(p) for p in params_partition)
        if len(self.in_axis) > 0:
            in_axis_argnums = [
                (x, v) for x, v in enumerate(self.in_axis) if v is not None
            ]
            nsplits = args[in_axis_argnums[0][0]].shape[
                in_axis_argnums[0][1]
            ]
        else:
            nsplits = next(iter(params_copy[0].values())).shape[0]

        for rkg in params_copy[0].filter(_(_RKGState)):
            rkg.value = rkg.split(nsplits)

        output, params_partition = self.transform(
            params_copy,
            *args
        )
        for p in params_partition[0]:
            p.reduce()

        return output

    def _make_transform(self):
        def vmap(
            *args,
            **kwargs
        ):
            outputs = self.f(*args, **kwargs)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            return jt.tree_map(
                lambda r, o: self._reduce(r, o, self.axis_name) if isinstance(o, str) else r,
                outputs, self.out_axis,
                is_leaf=lambda x: x is None
            )

        return jax.vmap(
            self._functional(vmap),
            in_axes=((0, None),) + self.in_axis,
            out_axes=(jt.tree_map(
                lambda o: 0 if isinstance(o, int) else None,
                self.out_axis,
                is_leaf=lambda r: r is None
            ), (0, None)),
            axis_name=self.axis_name,
        )

    @staticmethod
    def _reduce(x, mode, axis):
        if mode == "mean":
            return jax.lax.pmean(x, axis)
        elif mode == "sum":
            return jax.lax.psum(x, axis)


class _DerivativeBase(Transform):
    """Base class for various modules which compute derivatives."""

    def __init__(
        self,
        f: Union[Module, Callable],
        filter: Union[_, Callable[[ParamsDict], ParamsDict]],
        derivative_fn: Callable,
        input_argnums: Optional[Tuple[int, ...]] = None,
        has_aux: bool = False,
    ):
        super().__init__(f, filter)

        self.derivative_fn = derivative_fn
        self.input_argnums = input_argnums or tuple()
        self.has_aux = has_aux

    @property
    def partition(self) -> Tuple[ParamsDict, ParamsDict]:
        target = self.params.filter(self.filter)

        return target, self.params - target

    def _call(self, params_partition, *args, **kwargs):
        params_copy = tuple(move(p) for p in params_partition)
        inputs = [args[i] for i in self.input_argnums]

        g, (output, params_partition) = self.transform(
            (inputs, params_copy[0]),
            params_copy[1],
            *args,
            **kwargs,
        )
        # Map the gradients to the variables.
        g = (g[0], {id(k): v.value for k, v in zip(params_partition[0], g[1])})

        # Discard the input gradients if empty.
        if len(self.input_argnums) == 0:
            g = g[1]

        if self.has_aux:
            return (g, output)
        else:
            return g

    def _make_transform(self):
        def derivative(params_inputs, params_other, *args, **kwargs):
            inputs, params_target = params_inputs
            for i, arg in zip(self.input_argnums, inputs):
                args[i] = arg

            output, params_partition = self._functional(self.f)((params_target, params_other), *args, **kwargs)

            if not isinstance(output, tuple | list):
                output = (output,)

            return output[0], (output, params_partition)

        return self.derivative_fn(derivative)


class GradValues(_DerivativeBase):
    """The GradValues module is used to compute the gradients of a function."""

    def __init__(
        self,
        f: Union[Module, Callable],
        filter: Union[_, Callable[[ParamsDict], ParamsDict]],
        input_argnums: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__(
            f=f,
            filter=filter,
            derivative_fn=lambda func: jax.grad(func, has_aux=True),
            input_argnums=input_argnums,
            has_aux=True,
        )

        signature = inspect.signature(f)
        self.__signature__ = signature.replace(
            return_annotation=Tuple[
                Union[
                    Tuple[List[jax.Array], Dict[int, jax.Array]], Dict[int, jax.Array]
                ],
                signature.return_annotation,
            ]
        )
