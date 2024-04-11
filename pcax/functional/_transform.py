from typing import Any, Callable, Tuple, Sequence
from jaxtyping import PyTree
import abc
import inspect

import jax
import jax.tree_util as jtu
import equinox as eqx

from ..core._tree import tree_extract, tree_inject, tree_ref, tree_unref
from ..core._random import RKG
from ..core._parameter import BaseParam


########################################################################################################################
#
# TRANSFORM
#
# pcax keeps track of the changes applied to any Parameter. To achieve so, it introduces its own set of transformations
# which replace the ones provided by jax (such as jit, vmap, ...). Each transformation behaves exactly as its jax
# counterpart and is a simple wrapper that allows for parameter tracking and introduces some very small QOL improvements
# over the jax version.
#
# In particular, the protocol is defined as treating any positional argument as a "pure" jax one (i.e., not tracked),
# while it introduces the possibility to also use keyword arguments for transformations, whose Parameters are instead
# tracked. The most common usage pattern is to pass (stateful) models as keyword arguments, while any simple jax.Array
# as positional argument. For example:
#
# ```
# @vmap(...)
# def eval(x: jax.Array, y: jax.Array, *, model: pcax.Module):
#   y_hat = model(x)  # if model has any stateful layer such as batch norm, __call__ can safely update it
#   return ((y_hat - y)**2).mean()  # note that we do not return model, but its parameters are kept updated nontheless.
# ```
#
# NOTE #1: since pcax.Module is a pytree, one can safely pass it as positional argument as well, however its parameters
# will not be tracked.
#
# NOTE #2: pcax transformations automatically deal with multiple references to the same Parameter within the given
# keyword arguments (important: only for Parameters!). This enables parameter sharing.
#
# NOTE #3: all of this, requires each jax transformation to be ported to pcax. The parent class _BaseTransformation
# significantly automates this process but ad-hoc fixes are necessary for each function. In particular each pcax
# transformation must inherit from it and define its own _t function that, given the function self.fn with signature
#
# ```
# def fn(*args, **kwargs) -> *, params
# ```
#
# has the following signature:
#
# ```
# def _t(*args, **kwargs) -> *, params
# ```
#
# The standard way to do so is to define a _wrap_fn to wrap self.fn and rearrange input and output arguments to match
# the required structure.
#
# NOTE #4: some transformations require a kwargs_mask, which is an extra argument compared to the jax interface. It is
# necessary to specify which parameters in the kwargs must be affected by the transformation. Check
# '_AbstractTransform._process_mask' for a more detailed explanation.
#
########################################################################################################################


# Utils ################################################################################################################


def _make_tuple(x: Any):
    return x if isinstance(x, tuple) else (x,)


def _repr_function(f: Callable) -> str:
    """Human readable function representation."""
    _signature = inspect.signature(f)
    _args = [f"{k}={v.default}" for k, v in _signature.parameters.items() if v.default is not inspect.Parameter.empty]
    _args = ", ".join(_args)
    while not hasattr(f, "__name__"):
        if not hasattr(f, "func"):
            break
        f = f.func
    if not hasattr(f, "__name__") and hasattr(f, "__class__"):
        return f.__class__.__name__
    if _args:
        return f"{f.__name__}(*, {_args})"
    return f.__name__


# Core #################################################################################################################


class _BaseTransform(abc.ABC):
    """
    Base class for all transformations in pcax that wraps a jax transformation. __call__ keeps track of the parameters.
    Each derived class needs to only define a _t function with signature _t(*args, **kwargs) -> *, params. _t must wrap
    the target jax transformation and define all the necessary rearrangements of input and output arguments to use it.
    """

    def __init__(
        self,
        fn: "_BaseTransform" | Callable | Sequence["_BaseTransform" | Callable],
    ) -> None:
        """_BaseTransform constructor.

        Args:
            fn (_BaseTransform&#39; | Callable | Sequence[&#39;_BaseTransform&#39;  |  Callable]): the
                function (or sequence of function) to which the transformation is applied. As transformation can be
                composed, fn can be itself an '_BaseTransform'.
        """
        _fns = _make_tuple(fn)
        _fn = _fns[0]

        # We wrap 'fn' to return the updated values of the parameters as well and to unref the kwargs before passing
        # them to the original function.
        if isinstance(_fn, _BaseTransform):
            self.__wrapped__ = _fn.__wrapped__

            def _map_fn(fn):
                # 'kwargs' are reffed so we specify is_pytree=True to avoid reffing multiple times them.
                # This is a small optimization; in general, we could unref (the input 'kwargs')/ref (inside '__call__')
                # every time we call fn. Instead, we ref only once at the beginning of the nested transformation chain
                # and unref only at the end, when the original function is called.
                def _wrap_fn(*args, **kwargs):
                    # Update the global RKG key with the transformed one so it is accessible globally.
                    _old_key, RKG.key = RKG.key, kwargs["__RKG"].key
                    _r = fn(*args, **kwargs, _is_root=False)

                    # Replace the new key with the old ones to avoid leaks. It will be overwritten anyway immediately
                    # outside of the transformation bounds.
                    RKG.key = _old_key

                    return _r, kwargs

                return _wrap_fn

        else:
            self.__wrapped__ = _fn

            def _map_fn(fn):
                # We unref only when the original function, not a _BaseTransform, is called, as many transformations
                # could be nested one into the other, and it is necessary to unref only when the function is actually
                # called.
                def _wrap_fn(*args, **kwargs):
                    _old_key, RKG.key = RKG.key, kwargs["__RKG"].key
                    _fn_kwargs = tree_unref(kwargs)
                    del _fn_kwargs["__RKG"]
                    _r = fn(*args, **_fn_kwargs)

                    RKG.key = _old_key

                    return _r, kwargs

                return _wrap_fn

        self.fn = tuple(_map_fn(fn) for fn in _fns)
        if len(self.fn) == 1:
            self.fn = self.fn[0]

    def __call__(self, *args, _is_root: bool = True, **kwargs: Any) -> Any:
        """Call the transformed function.

        Args:
            _is_root (bool, reserved): used to distinguish between recursive calls.

        Returns:
            Any: the transformed output of the wrapped function.
        """

        # Inject the default RKG
        if "__RKG" not in kwargs:
            kwargs["__RKG"] = RKG

        if _is_root is True:
            kwargs = tree_ref(kwargs)
        _r, _kwargs = self._t(*args, **kwargs)

        # This is the key part: the updated values are injected back into the original parameters in 'kwargs'. 'kwargs'
        # is still the original structure as it hasn't undergone any transformation (which happens only inside '_t').
        # The update values are obtained by calling 'extract', which is done automatically by the wrapped 'fn'.
        # If 'tree_extract' is called before returning '_kwargs', a list of value was returned, so we tell 'tree_inject'
        # to handle it correctly.
        if isinstance(kwargs, dict):
            tree_inject(kwargs, params=_kwargs, is_pytree=True)
        else:
            tree_inject(kwargs, values=_kwargs, is_pytree=True)

        return _r

    @abc.abstractmethod
    def _t(self, *args, **kwargs) -> Tuple[Any, Sequence[jax.Array | None]]:
        return NotImplemented

    def __repr__(self):
        _fn = (
            repr(self.__wrapped__) if isinstance(self.__wrapped__, _BaseTransform) else _repr_function(self.__wrapped__)
        )
        return f"{self.__class__.__name__}(fn={_fn})"

    @staticmethod
    def _process_mask(mask: PyTree, kwargs: PyTree, rkg_mask=None) -> PyTree:
        """Applies the mask to the given kwargs. If the mask keys are tuples, they are expanded into individual keys.
        This utility is provided as several jax transformations require a mask to know which jax.Arrays to target.

        Args:
            mask (PyTree): a pytree with the same structure as kwargs, or a valid prefix of it, whose leaves are either
                callable objects or masked value. If a callable object is given, then it is applied to the corresponding
                kwarg subtree and the result is used as the mask. If a masked value is given, then it is used as mask.
            kwargs (PyTree): keyword arguments to which the mask is applied.

        Returns:
            PyTree: masked keyword arguments.
        """
        if isinstance(mask, dict):
            mask = {k_i: v for k, v in mask.items() for k_i in _make_tuple(k)}

            # Inject RKG mask
            mask["__RKG"] = rkg_mask

        def map_fn(mask, kwarg):
            if callable(mask):
                try:
                    # special case handling for Mask (or any callable that may benefit by knowing it is being
                    # called on a guaranteed pytree), since 'kwarg' is already reffed, and thus a pytree.
                    return mask(kwarg, is_pytree=True)
                except TypeError:
                    return mask(kwarg)
            else:
                return mask

        # kwargs is reffed so we don't have to worry about (un)flattening its parameters,
        # so no need to specify a is_leaf function.
        mask = jtu.tree_map(map_fn, mask, kwargs)

        # In case the custom mask overwrites the __RKG mask
        mask["__RKG"] = rkg_mask

        return mask


class Jit(_BaseTransform):
    """
    Wrap around jax.jit(fn, ...).

    Uses 'tree_extract' to return a list of parameters instead of a complex pytree.
    This is used to reduce the overhead of injecting the new values back into the
    original kwargs outside of the "jit barrier".
    """

    def __init__(self, fn: "_BaseTransform" | Callable, **t_kwargs: Any):
        super().__init__(fn)

        def _wrap_fn(*args, **kwargs):
            _r, _kwargs = self.fn(*args, **kwargs)

            return _r, tree_extract(_kwargs, is_pytree=True)

        self.wrap_fn = jax.jit(_wrap_fn, **t_kwargs)

    def _t(self, *args, **kwargs):
        _r, kwargs = self.wrap_fn(*args, **kwargs)

        return _r, kwargs


class ValueAndGrad(_BaseTransform):
    """
    Wrap around jax.value_and_grad(fn, ...).
    kwargs_mask must specify whether each leaf is to be differentiated (True) or not (False).

    NOTE #1: the optimizer class provided by pcax assumes that each 'BaseParam' is a leaf of the optimized pytree,
    this implies that the mask must provide a value at the 'BaseParam' level and not for its value. For example:

    ```python

    model = [Param(jax.numpy.array([1.0])), Param(jax.numpy.array([2.0]))]
    mask = [True, False]  # this is correct, [Param(True), Param(False)] would be wrong.
    ```

    This is the behavior of 'Mask' which replaces each parameter (and not the value of the parameters) with the given
    values.

    NOTE #2: #1 implies that it is not possible to differentiate the behavior for different 'jax.Arrays' within a single
    parameter (in general, the value of a parameter can be anything, and it is not limited to a single 'jax.Array', see
    for example the 'ParamDict' class). So please keep this in mind when using 'ValueAndGrad' (or any other transform
    that requires a mask) and structure your models accordingly (i.e., different parameters for different behaviours).

    NOTE #3: the assumption in #1 is to simplify the implementation of the optimizer class and in general it seemed
    more intuitive and less error prone (i.e., masking substitutes the whole parameter object, and not only its value),
    and it also required to completly discard static parameters (which, otherwise, would clutter the mask with
    unnecessary static information which could cause pytree incompatibilities down the line, which is what would indeed
    happen in the optimizer). Furthermore, it allows to deal with unreffed pytrees (maybe, to be verfied with the
    behaviour of each jax transformation). However, the assumption it is not a strict requirement and the code can
    easily be redesigned to allow for masking of the value if deemed to be a necessary feature.
    """

    def __init__(self, fn: "_BaseTransform" | Callable, kwargs_mask: Any = {}, **t_kwargs: Any):
        super().__init__(fn)
        self.kwargs_mask = kwargs_mask
        self.has_aux = t_kwargs["has_aux"]

        t_kwargs["has_aux"] = True
        self.t_kwargs = t_kwargs

    def _t(self, *args, **kwargs):
        def _wrap_fn(*args):
            _args, _target_kwargs, _other_kwargs = args[:-2], args[-2], args[-1]
            _kwargs = eqx.combine(_target_kwargs, _other_kwargs, is_leaf=lambda x: isinstance(x, BaseParam))
            _r, _kwargs = self.fn(*_args, **_kwargs)
            _r = _make_tuple(_r)

            return _r[0], (_r[1:], _kwargs)

        (_l, (_r, _values)), _aux = jax.value_and_grad(
            _wrap_fn, **{**self.t_kwargs, "argnums": self.t_kwargs.get("argnums", ()) + (len(args),)}
        )(
            *args,
            # We split kwargs to isolate the parameters we want to differentiate, following the jax syntax.
            *eqx.partition(
                kwargs,
                # we pass 'False' as rkg mask to not take its gradient.
                self._process_mask(self.kwargs_mask, kwargs, False),
                is_leaf=lambda x: isinstance(x, BaseParam),
            ),
        )

        if self.t_kwargs.get("argnums", ()) != ():
            _aux = (_aux[:-1], _aux[-1])
        else:
            _aux = _aux[0]

        if self.has_aux:
            return (((_l, _r), _aux), _values)
        else:
            return ((_l, _aux), _values)


class Vmap(_BaseTransform):
    """
    Wrap around jax.vmap(fn, ...).
    kwargs_mask must specify whether each leaf is vectorised or not. It is assumed that the behaviour for
    each leaf is the same for both input and output (this could be changed by providing an 'out_kwargs_mask'
    as well as a 'in_kwargs_mask'). Both 'in_axes' and 'out_axes' must be provided in a jax supported format.

    NOTE: RKG is automatically handled by the transformation, so it must not be provided in the kwargs.
    """

    def __init__(self, fn: "_BaseTransform" | Callable, kwargs_mask: Any = {}, **t_kwargs: Any):
        super().__init__(fn)
        self.kwargs_mask = kwargs_mask
        self.t_kwargs = t_kwargs

    def _t(self, *args, **kwargs):
        _kwargs_mask = self._process_mask(self.kwargs_mask, kwargs)
        _in_axes_mask = _make_tuple(self.t_kwargs.get("in_axes", ())) + (_kwargs_mask,)

        # Compute vaxes dimension which is necessary to split the RKG key.
        def _extract_vaxes_dim(node, mask):
            for param in filter(lambda _node: hasattr(_node, "shape"), jtu.tree_leaves(node)):
                return param.shape[mask]

            return None

        _vaxis_dim = jtu.tree_leaves(
            jtu.tree_map(lambda mask, node: _extract_vaxes_dim(node, mask), _in_axes_mask, (*args, kwargs))
        )[0]

        # Split the __RKG key over the vmap axis (and set the mask accordingly)
        _in_axes_mask[-1]["__RKG"] = 0
        kwargs["__RKG"].key.set(kwargs["__RKG"].key.split(_vaxis_dim))

        def _wrap_fn(*args):
            *_args, _kwargs = args
            _r, _kwargs = self.fn(*_args, **_kwargs)

            return _r, _kwargs

        _r, kwargs = jax.vmap(
            _wrap_fn,
            **{
                **self.t_kwargs,
                "in_axes": _in_axes_mask,
                "out_axes": (self.t_kwargs.get("out_axes", None), _kwargs_mask),
            },
        )(*args, kwargs)

        # Merge back the key value to remove the vmap axis before returning it;
        # it will automatically be injected back into the global RKG (being it a kwarg)
        kwargs["__RKG"].key.set(kwargs["__RKG"].key[0])

        return _r, kwargs
