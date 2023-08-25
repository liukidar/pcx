__all__ = [
    "_AbstractTransformation",
    "Jit",
    "Vectorize",
    "GradAndValues",
]

import abc
import copy
import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.tree_util as jt

from pcax.core.filter import f

from ..core.modules import Function, Module
from ..core.parameters import ParamDict
from ..core.random import RKG, _RKGState
from ..core.util import hash_pytree, move, repr_function

########################################################################################################################
#
# TRANSFORMATIONS
#
# pcax offers a wrapper around the most important jax transformations, such as vmap and value_and_grad. A wrapper is
# necessary in order to catch the tensors used within the transformation and provide an imperative approach to the user.
# Furthermore, the classes defined here offer some quality-of-life improvement that facilitate their usage.
# NOTE: Compared to JAX, some functionalities and flexibility are missing/not accessible to the user. Improving this
# aspect is a core issues that will be addressed in future versions.
#
########################################################################################################################


class _AbstractTransformation(abc.ABC):
    """
    Base abstract class for all transformations. It is used to track the parameters belonging to any module passed to
    it. By default, it also keeps track of the default random key generator pcax.RKG.
    Transformations are lazily computed on the fly when calling them by analyzing the kwargs of the transformed
    function. In particular, args passed to a function are considered dynamic (i.e., jax.Array), while kwargs are
    considered static. Modules *must* be passed as a kwarg to be registered to the transformation. It is possible to
    pass them as args but they will be treated as normal pytrees of parameters (e.g., it would be similar to passing
    a dictionary of tensors to the jax transformation).
    """

    def __init__(
        self,
        fn: Union["_AbstractTransformation", Callable],
        filter: Union[f, Callable[[ParamDict], ParamDict]],
    ):
        """_AbstractTransformation constructor.

        Args:
            f: the function (or pcax transformation) to be processed by the current transformation,
            filter: the filter used to select which parameters should undergo the transformation.
        """
        if isinstance(fn, _AbstractTransformation):
            self.__wrapped__ = fn.__wrapped__
            self.fn = copy.deepcopy(fn)
        else:
            self.__wrapped__ = fn
            self.fn = fn
        self.params: ParamDict = None
        self.filter = filter
        self.transform = None
        self.kwargs = {}

    def __call__(self, *args, **kwargs: Any) -> Any:
        fn = self._build(
            functools.reduce(
                lambda x, y: x + y,
                (
                    m.parameters().rename(k)
                    for k, m in kwargs.items()
                    if isinstance(m, Module)
                ),
                RKG.parameters(),
            ),
            kwargs,
        )
        return fn(*args)

    def _build(self, params, kwargs):
        if isinstance(self.fn, _AbstractTransformation):
            t = self.fn._build(params, kwargs)
        else:
            t = self.fn

        self.params = params

        self.transform = self._make_transform(t, kwargs)

        def fn(*args):
            return self._call(self.partition, *args)

        return Function(fn, self.params)

    def _functional(self, t: Callable, kwargs: Dict[str, Any]) -> Callable:
        def wrapper(params_copy, *args):
            params_partition = tuple(
                move(c, p) for c, p in zip(params_copy, self.partition)
            )
            if isinstance(t, Function):
                output = t(*args)
            else:
                output = t(*args, **kwargs)

            return output, tuple(move(p) for p in params_partition)

        return wrapper

    def update_partition(self, new_partition):
        old_target, old_params = self.partition
        new_target, new_params = new_partition

        move(new_target, old_target)
        move(new_params, old_params)

    @property
    def partition(self) -> Tuple[ParamDict, ParamDict]:
        target = self.params.filter(self.filter) + RKG.parameters()

        return target, self.params - target

    @abc.abstractmethod
    def _call(self, *args):
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_transform(self, fn, kwargs):
        raise NotImplementedError()

    def __repr__(self):
        fn = (
            repr(self.__wrapped__)
            if isinstance(self.__wrapped__, _AbstractTransformation)
            else repr_function(self.__wrapped__)
        )
        return f"{self.__class__.__name__}(fn={fn})"


class Jit(_AbstractTransformation):
    """pcax transformation corresponding to jax.jit.

    Differently from other transformations, we may want not to compute a jitted function on the fly for two reasons:
    - we want to jit a function and reuse it many times to benefit from the compilation.
    - even when using the caching functionality integrated within pcax.jit, we actually waste computing resources to
    compute the function's arguments hash to determine if a cached version of such function already exists.
    pcax gives the user the possibility to manually handle this step, saving computing at the expense of requiring
    more carefulness when using such functionality.

    In particular, it is possible to create `snapshots` of a to-be-jitted function fn. A snapshot will create a compiled
    version of fn with the current kwargs passed to it and will ignore any further changes made to them (i.e., skip the
    cache hit/miss bit and always use the same jitted function, regardless of changes in it inputs). It is the user who
    will have to keep track of any possible change in the kwargs structure and create new snapshots when needed.

    Example:

    @pcax.jit()
    def fn(x, y, model, optimizer):
        g = loss(x, y, model=model)
        optimizer.step(g)

    # The following two ways produce the same computation, however one will keep track of changes in the static
    # arguments requiring extra computations

    #
    # Default mode: the hash of model and optimizer is computed, this is a rather expensive computation, but changes in
    # the structure are possible
    #
    fn(x, y, model=model, optimizer=optimizer)
    fn(x, y, model=model, optimizer=optimizer) # this has to compute the hash of model and optimizer again
    model.layer1 = new Layer(...)  # different parameters compared to the previous layer1
    fn(x, y, model=model, optimizer=optimizer) # this will correctly compute a new computational graph

    #
    # Snapshot mode: the hash of model and optimizer is cached and reused. Changes in the structure are not tracked and
    # will result in undefined behaviour.
    #
    fn_s = fn.snapshot(model=model, optimizer=optimizer)
    fn_s(x, y) # no kwargs are passed to the snapshot as those are cached
    fn_s(x, y) # fast re-execution: no hash is computed
    model.layer1 = new Layer(...)  # different parameters compared to the previous layer1
    fn_s(x, y) # this is undefined behaviour
    fn_s = fn.snapshot(model=model, optimizer=optimizer) # need to manually recompute the snapshot
    fn_s(x, y) # now you can use it
    """

    def __init__(
        self,
        fn: Union[_AbstractTransformation, Callable],
        filter: Union[f, Callable[[ParamDict], ParamDict]] = lambda *args: True,
        donate_argnums: Tuple[int, ...] = (),
        inline: bool = False,
    ):
        """Jit constructor.

        Args:
            fn: the function/transformation to jit,
            filter: the filter used to select which parameters should be tracked by the jitter
                (by default is all of them),
            donate_argnums: same as jax.jit (only affects args),
            inline: same as jax.jit,
        """

        super().__init__(fn, filter)

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
                (
                    m.parameters().rename(k)
                    for k, m in kwargs.items()
                    if isinstance(m, Module)
                ),
                RKG.parameters(),
            ),
            kwargs,
        )

    def _call(self, params_partition, *args):
        output, new_partition = self.transform(
            params_partition, self.static_args_hash, *args
        )
        self.update_partition(new_partition)

        return output

    def _build(self, params, kwargs):
        self.static_args_hash = hash_pytree(kwargs)

        return super()._build(params, kwargs)

    def _make_transform(self, fn, kwargs):
        return jax.jit(
            lambda params, _, *args: self._functional(fn, kwargs)(params, *args),
            static_argnums=(1,),
            donate_argnums=(0,) + tuple(x + 2 for x in self.donate_argnums),
            inline=self.inline,
        )


class Vectorize(_AbstractTransformation):
    """pcax transformation corresponding to jax.vmap.

    Compared to jax.vmap it does not yet support complex axis selection for vmapping. However, it offers
    automatic reduction of the output. Supported reduction modes are: 'mean', 'sum'; which can be selected by passing
    such string to the corresponding `out_axis` element, instead of a number specifing the vmapped axis.
    """

    def __init__(
        self,
        fn: Union[Module, Callable],
        filter: Union[f, Callable[[ParamDict], ParamDict]],
        in_axis: Tuple[Optional[int], ...] = (0,),
        out_axis: Tuple[Optional[int], ...] = (0,),
        axis_name: str = "batch",
    ):
        """Vectorize constructor.

        Args:
            fn: the function/transformation to vectorize,
            filter: the filter used to select which parameters should be vectorized,
            in_axis: same as jax.vmap (only affects args),
            out_axis: same as jax.vmap, but can also contain reduction modes,
            axis_name: same as jax.vmap.
        """

        super().__init__(fn, filter)

        self.in_axis = in_axis
        self.out_axis = out_axis
        self.axis_name = axis_name

    def _call(self, params_partition, *args):
        if len(self.in_axis) > 0:
            in_axis_argnums = [
                (x, v) for x, v in enumerate(self.in_axis) if v is not None
            ]
            nsplits = args[in_axis_argnums[0][0]].shape[in_axis_argnums[0][1]]
        else:
            nsplits = next(iter(params_partition[0].values())).shape[0]

        for rkg in params_partition[0].filter(f(_RKGState)):
            rkg.value = rkg.split(nsplits)

        output, new_partition = self.transform(params_partition, *args)
        for p in new_partition[0]:
            p.reduce()

        self.update_partition(new_partition)

        return output

    def _make_transform(self, fn, kwargs):
        def vmap(*fn_args, **fn_kwargs):
            outputs = fn(*fn_args, **fn_kwargs)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            return jt.tree_map(
                lambda r, o: self._reduce(r, o, self.axis_name)
                if isinstance(o, str)
                else r,
                outputs,
                self.out_axis,
                is_leaf=lambda x: x is None,
            )

        return jax.vmap(
            self._functional(vmap, kwargs),
            in_axes=((0, None),) + self.in_axis,
            out_axes=(
                jt.tree_map(
                    lambda o: 0 if isinstance(o, int) else None,
                    self.out_axis,
                    is_leaf=lambda r: r is None,
                ),
                (0, None),
            ),
            axis_name=self.axis_name,
        )

    @staticmethod
    def _reduce(x, mode, axis):
        if mode == "mean":
            return jax.lax.pmean(x, axis)
        elif mode == "sum":
            return jax.lax.psum(x, axis)


class _DerivativeBase(_AbstractTransformation):
    """Base class for various modules which compute derivatives. Currently used only by pcax.GradValues."""

    def __init__(
        self,
        fn: Union[Module, Callable],
        filter: Union[f, Callable[[ParamDict], ParamDict]],
        derivative_fn: Callable,
        input_argnums: Tuple[int, ...] = (),
        has_aux: bool = False,
    ):
        """_DerivativeBase constructor.

        Args:
            - fn: the function/transformation to which to apply `derivative_fn`,
            - filter: the filter used to select which parameters should be targeted by `derivative_fn`,
            - derivative_fn: the jax derivative transformation to use,
            - input_argnums: indices of the input arguments to be targeted by `derivtive_fn` (default None),
            - has_aux: whether derivative_fn returns an auxiliary value.
        """
        super().__init__(fn, filter)

        self.derivative_fn = derivative_fn
        self.input_argnums = input_argnums
        self.has_aux = has_aux

    @property
    def partition(self) -> Tuple[ParamDict, ParamDict]:
        target = self.params.filter(self.filter)

        return target, self.params - target

    def _call(self, params_partition, *args):
        params_copy = tuple(move(p) for p in params_partition)
        inputs = [args[i] for i in self.input_argnums]

        g, (output, new_partition) = self.transform(
            (inputs, params_copy[0]), params_copy[1], *args
        )
        # Move new_partition into self.params before mapping the gradients to the variables.
        self.update_partition(new_partition)

        # Map the gradients to the variables.
        g = (g[0], {id(k): v.value for k, v in zip(params_partition[0], g[1])})

        # Discard the input gradients if empty.
        if len(self.input_argnums) == 0:
            g = g[1]

        if self.has_aux:
            return (g, output)
        else:
            return g

    def _make_transform(self, fn, kwargs):
        def derivative(params_inputs, params_other, *args):
            inputs, params_target = params_inputs
            for i, arg in zip(self.input_argnums, inputs):
                args[i] = arg

            output, params_partition = self._functional(fn, kwargs)(
                (params_target, params_other), *args
            )

            if not isinstance(output, tuple | list):
                output = (output,)

            return output[0], (output, params_partition)

        return self.derivative_fn(derivative)


class GradAndValues(_DerivativeBase):
    """pcax transformation corresponding to jax.value_and_grad.
    The output gradients are returned according to the following specification:
    - if any input gradient is present: (g_wrt_inputs, g_wrt_parameters),
    - if no input gradient is requested: g_wrt_parameters,
    where:
    - g_wrt_inputs is a list containing the gradients wrt each input, ordered by `input_argnums`,
    - g_wrt_parameters is a dictionary where each element is pair (key_p, g_wrt_p), with p being any of the target
    parameters, key_p = id(p), and g_wrt_p is the gradient wrt to p.value (`id` is a python predefined function).
    """

    def __init__(
        self,
        fn: Union[Module, Callable],
        filter: Union[f, Callable[[ParamDict], ParamDict]],
        input_argnums: Tuple[int, ...] = (),
    ):
        super().__init__(
            fn=fn,
            filter=filter,
            derivative_fn=lambda func: jax.grad(func, has_aux=True),
            input_argnums=input_argnums,
            has_aux=True,
        )

        signature = inspect.signature(fn)
        self.__signature__ = signature.replace(
            return_annotation=Tuple[
                Union[
                    Tuple[List[jax.Array], Dict[int, jax.Array]], Dict[int, jax.Array]
                ],
                signature.return_annotation,
            ]
        )
