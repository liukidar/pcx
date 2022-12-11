__all__ = [
    "Jit",
    "Vectorize",
    "GradValues",
]

from typing import Optional, List, Union, Callable, Tuple, Any, Dict
import inspect
import functools

import jax
import jax.tree_util as jt
from objax.util import class_name, repr_function

from ..core.filter import _
from ..core.structure import Module, Function, RandomState, VarCollection
from ..core.random import DEFAULT_GENERATOR
from ..core.util import positional_args_names


class ModuleTransform(Module):
    def __init__(
        self,
        f: Union[Module, Callable],
        vc_f: Union[_, Callable[[VarCollection], VarCollection]],
    ):
        if not isinstance(f, Module):
            f = Function(f)

        self.vc = f.vars() + VarCollection(DEFAULT_GENERATOR.vars())
        self.vc_f = (lambda vc: vc.filter(vc_f)) if isinstance(vc_f, _) else vc_f
        self.__wrapped__ = f

    def _functional(self, f: Callable) -> Callable:
        """Return a functional version of f."""

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            original = self.vc.dump()[:2]
            try:
                return f(*args, **kwargs)
            finally:
                self.vc.load(original)

        return wrapper

    def vars(self, scope: str = "") -> VarCollection:
        """Return the VarCollection of the variables used."""
        if scope:
            return VarCollection((scope + k, v) for k, v in self.vc.items())
        return VarCollection(self.vc)


class Jit(ModuleTransform):
    """JIT (Just-In-Time) module takes a function or a module and compiles it for faster execution."""

    def __init__(
        self,
        f: Union[Module, Callable],
        vc_f: Union[_, Callable[[VarCollection], VarCollection]] = lambda vc: vc,
        static_argnums: Optional[Tuple[int, ...]] = None,
    ):
        """Jit constructor.

        Args:
            f: the function or the module to compile.
            vc: the VarCollection of variables used by the function or module. This argument is required for functions.
            static_argnums: tuple of indexes of f's input arguments to treat as static (constants)).
                A new graph is compiled for each different combination of values for such inputs.
        """
        super().__init__(f, vc_f)
        self.static_argnums = static_argnums

        @self._functional
        def jit(
            params: Tuple[Tuple[jax.Array, ...], Tuple[Any, ...]],
            static: Tuple[Any, ...],
            kwargs: Dict[str, Any],
            *args,
        ):
            self.vc_target.load(*params)
            return f(*args, **kwargs), self.vc.dump()[:2]

        self._call = jax.jit(
            jit,
            static_argnums=(1,) + tuple(x + 3 for x in sorted(static_argnums or ())),
        )

    @property
    def vc_target(self) -> VarCollection:
        return self.vc_f(self.vc) + VarCollection(DEFAULT_GENERATOR.vars())

    def __call__(self, *args, **kwargs):
        """Call the compiled version of the function or module."""
        differentiable, dynamic, static = self.vc_target.dump()

        """Pass static to keep track of changes in static variables."""
        output, changes = self._call(
            (differentiable, dynamic), hash(static), kwargs, *args
        )
        self.vc.load(*changes)

        return output

    def __repr__(self):
        return f"{class_name(self)}(f={self.__wrapped__}, static_argnums={self.static_argnums or None})"


def _reduce(x, mode, axis):
    if mode == "mean":
        return jax.lax.pmean(x, axis)
    elif mode == "sum":
        return jax.lax.psum(x, axis)


class Vectorize(ModuleTransform):
    """Vectorize module takes a function or a module and compiles it for running in parallel on a single device."""

    def __init__(
        self,
        f: Union[Module, Callable],
        vc_f: Union[_, Callable[[VarCollection], VarCollection]],
        in_axis: Tuple[Optional[int], ...] = (0,),
        out_axis: Tuple[Optional[int], ...] = (0,),
        axis_name: str = "batch",
    ):
        """Vectorize constructor.

        Args:
            f: the module to compile for vectorization.
            vc: the VarCollection of variables used by f (can be a function to select a subset from f).
            in_axis: tuple of int or None for each of f's input arguments: the axis to use as batch during
                vectorization. Use None to automatically broadcast.
            out_axis: tuple of int or None for each of f's output arguments: the axis to use as batch during
                vectorization. Use None if no broadcast is necessary.
            axis_name: what name to give to the batch dimension.
        """
        super().__init__(f, vc_f)

        @self._functional
        def vmap(
            vmap_params: List[jax.Array],
            random_list: List[jax.Array],
            *args,
        ):
            self.vc_target.load(vmap_params)
            for state, value in zip(self.vc_target.filter(_(RandomState)), random_list):
                state.value = value

            outputs = f(*args)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            return (
                jt.tree_map(
                    lambda r, o: _reduce(r, o, axis_name) if isinstance(o, str) else r,
                    outputs, out_axis,
                    is_leaf=lambda r: r is None
                ),
                self.vc.dump()[:2],
            )

        fargs = positional_args_names(f)
        assert len(in_axis) >= len(
            fargs
        ), f"The batched argument must be specified for all of {f} arguments {fargs}"
        self.in_axis = in_axis
        self.in_axis_argnums = [
            (x, v) for x, v in enumerate(in_axis) if v is not None
        ]
        self._call = jax.vmap(
            vmap,
            in_axes=(0, 0) + in_axis,
            out_axes=(jt.tree_map(
                lambda o: 0 if isinstance(o, int) else None,
                out_axis,
                is_leaf=lambda r: r is None
            ), 0),
            axis_name=axis_name,
        )

    @property
    def vc_target(self) -> VarCollection:
        return self.vc_f(self.vc) + VarCollection(DEFAULT_GENERATOR.vars())

    def __call__(self, *args):
        """Call the vectorized version of the function or module."""
        assert len(args) == len(self.in_axis), (
            f"Number of arguments passed {len(args)} must match "
            f"batched {len(self.in_axis)}"
        )
        nsplits = args[self.in_axis_argnums[0][0]].shape[
            self.in_axis_argnums[0][1]
        ]
        output, changes = self._call(
            self.vc_target.dump()[0],
            [v.split(nsplits) for v in self.vc_target.filter(_(RandomState))],
            *args,
        )
        self.vc.load(*changes, reduce=True)

        return output

    def __repr__(self):
        return f"{class_name(self)}(f={self.__wrapped__}, in_axis={self.in_axis})"


class _DerivativeBase(ModuleTransform):
    """Base class for various modules which compute derivatives."""

    def __init__(
        self,
        derivative_fn: Callable,
        f: Union[Module, Callable],
        vc_f: Union[_, Callable[[VarCollection], VarCollection]],
        input_argnums: Optional[Tuple[int, ...]] = None,
        return_all_f_outputs: bool = False,
    ):
        """Constructs an instance to compute the derivatives of f w.r.t. variables.
        Args:
            derivative_fn: JAX transformation which computes derivative.
            f: the function for which to compute derivatives.
            variables: the variables for which to compute derivatives.
            input_argnums: input indexes, if any, on which to compute derivatives.
            return_all_f_outputs: if True also return original outputs of the fuction along with derivatives.
        """
        super().__init__(f, vc_f)

        self.input_argnums = input_argnums or tuple()
        self.return_all_f_outputs = return_all_f_outputs

        @self._functional
        def f_func(
            inputs_and_train_tensors: List[jax.Array],
            list_args: List,
            kwargs: Dict,
        ):
            inputs, train_tensors = inputs_and_train_tensors
            self.vc_target.load(train_tensors)

            for i, arg in zip(self.input_argnums, inputs):
                list_args[i] = arg
            outputs = f(*list_args, **kwargs)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            if self.return_all_f_outputs:
                return outputs[0], (outputs, self.vc.dump()[:2])
            else:
                return outputs[0], self.vc.dump()[:2]

        self._call = derivative_fn(f_func)

    @property
    def vc_target(self) -> VarCollection:
        return self.vc_f(self.vc)

    def __call__(self, *args, **kwargs):
        """Returns the computed gradients for the first value returned by `f` and optionally values returned by `f`."""
        inputs = [args[i] for i in self.input_argnums]

        g, aux_out = self._call(
            (inputs, self.vc_target.dump()[0]),
            list(args),
            kwargs,
        )

        # Map the gradients to the variables.
        g = (g[0], {id(k): v for k, v in zip(self.vc_target.values(), g[1])})

        # Discard the input gradients if empty.
        if len(self.input_argnums) == 0:
            g = g[1]

        if self.return_all_f_outputs:
            outputs, changes = aux_out
            self.vc.load(*changes)

            return g, outputs
        else:
            changes = aux_out
            self.vc.load(*changes)

            return g

    def __repr__(self):
        f = (
            repr(self.__wrapped__)
            if isinstance(self.__wrapped__, Module)
            else repr_function(self.__wrapped__)
        )
        return f"{class_name(self)}(f={f}, input_argnums={self.input_argnums or None})"


class GradValues(_DerivativeBase):
    """The GradValues module is used to compute the gradients of a function."""

    def __init__(
        self,
        f: Union[Module, Callable],
        vc_f: Union[_, Callable[[VarCollection], VarCollection]],
        input_argnums: Optional[Tuple[int, ...]] = None,
    ):
        """Constructs an instance to compute the gradient of f w.r.t. variables.

        Args:
            f: the function for which to compute gradients.
            variables: the variables for which to compute gradients.
            input_argnums: input indexes, if any, on which to compute gradients.
        """

        super().__init__(
            lambda func: jax.grad(func, has_aux=True),
            f=f,
            vc_f=vc_f,
            input_argnums=input_argnums,
            return_all_f_outputs=True,
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
