__all__ = ["Scan", "WhileLoop", "Cond", "Switch"]


from typing import Callable, Any, Sequence
from functools import partial

import jax

from ._transform import _make_tuple, _BaseTransform


########################################################################################################################
#
# FLOW
#
# Flow transformations are a set of transformations that are used to define the flow of control in a program since jax
# does not allow to use control flow statements (if, for, while, ...) with dynamic values (e.g., tensors).
# For example, the following code is not allowed in jax:
#
# ```python
# @jax.jit
# def f(x):
#     if x > 0:
#         return x
#     else:
#         return -x
# f(jax.numpy.array([1, -1]))  # NOTE: this raises an error
# ```
#
# Instead, we rely on the primitives in jax.lax (i.e., cond, scan, ...), which are wrapped by pcax to allow for
# automatic parameter tracking. Each transformation is a thin wrapper around the corresponding jax.lax function and
# behaves analogously to its jax counterpart. The main difference is that the input and output arguments require a
# slighlty different structure to simplify usage. Check the documentation of each transformation for more details.
# Furthermore, to be consistent with the _BaseTransformation class, each transformation accepts the function
# arguments via the __call__ method and not directly in the constructor (i.e., Scan(f, ...)(x), instead of
# scan(f, x, ...) as it is in jax). However, the utility functions provided at the end of this file allow to use the
# jax syntax as well.
########################################################################################################################


class Scan(_BaseTransform):
    """
    pcax wrap around ``jax.lax.scan(fn, ...).``
    Takes the same function arguments (except 'init') but does not require a compact carry argument.
    In particular, ``fn`` must have signature ``fn(x, *args, **kwargs) -> args, y``.

    Example:

    .. code-block:: python

        def f(x, count):
            count = count + x
            return (count + x,), None

        Scan(f, xs=jax.numpy.arange(5))(0)  # [0, 1, 3, 6, 10], None
    """

    def __init__(self, fn: "_BaseTransform" | Callable, **t_kwargs: Any):
        """Scan constructor.

        Args:
            fn (_BaseTransformation | Callable): function corresponding to `f` for jax.lax.scan
        """
        super().__init__(fn)

        self.t_kwargs = t_kwargs

    def _t(self, *args, **kwargs):
        def _wrap_fn(args, x):
            _args, _kwargs = args
            (_r, _y), _kwargs = self.fn(x, *_args, **_kwargs)

            return (_make_tuple(_r), _kwargs), _y

        (_r, kwargs), _y = jax.lax.scan(_wrap_fn, (args, kwargs), **self.t_kwargs)
        return (_r, _y), kwargs


class WhileLoop(_BaseTransform):
    """
    pcax wrap around jax.lax.while_loop(f, ...).
    Takes the same function arguments (except 'init_val') but does not require a compact val argument.
    In particular, fn must have signature fn(*args, **kwargs) -> args, while cond_fun has signature
    cond_fun(*args, **kwargs) -> bool.
    """

    def __init__(self, fn: _BaseTransform | Callable, **t_kwargs: Any):
        """WhileLoop constructor.

        Args:
            fn (_BaseTransformation | Callable): function corresponding to `body_fun`
                for jax.lax.while_loop.
        """
        super().__init__(fn)

        self.cond_fun = t_kwargs["cond_fun"]
        del t_kwargs["cond_fun"]
        self.t_kwargs = t_kwargs

    def _t(self, *args, **kwargs):
        def _wrap_fn(args):
            _args, _kwargs = args
            _r, _kwargs = self.fn(*_args, **_kwargs)

            return _make_tuple(_r), _kwargs

        # we allow `cond_fun` to look at both args and kwargs
        def _cond_fn(carry):
            args, kwargs = carry
            del kwargs["__RKG"]

            return self.cond_fun(*args, **kwargs)

        _r, kwargs = jax.lax.while_loop(
            _cond_fn, _wrap_fn, (args, kwargs), **self.t_kwargs
        )
        return _r, kwargs


class Cond(_BaseTransform):
    """
    pcax wrap around jax.lax.cond(..., true_fn, false_fn, ...).
    Takes the same function arguments (except 'cond' and '*operands').
    """

    def __init__(
        self,
        true_fn: _BaseTransform | Callable,
        false_fn: _BaseTransform | Callable,
        **t_kwargs: Any,
    ):
        """Cond constructor.

        Args:
            true_fn (_BaseTransformation | Callable): function to be called if pred is True.
            false_fn (_BaseTransformation | Callable): function to be called if pred is False.
        """
        super().__init__((true_fn, false_fn))

        self.t_kwargs = t_kwargs

    def _t(self, cond, *args, **kwargs):
        def _wrap_fn(i, args):
            _args, _kwargs = args
            _r, _kwargs = self.fn[i](*_args, **_kwargs)

            return _r, _kwargs

        _r, kwargs = jax.lax.cond(
            cond,
            *tuple(partial(_wrap_fn, i) for i in range(len(self.fn))),
            (args, kwargs),
            **self.t_kwargs,
        )
        return _r, kwargs


class Switch(_BaseTransform):
    """
    pcax wrap around jax.lax.switch(..., branches, ...).
    Takes the same function arguments (except 'index', and '*operands').
    """

    def __init__(self, fns: Sequence[_BaseTransform | Callable], **t_kwargs: Any):
        """Switch constructor.

        Args:
            fns (Sequence[_BaseTransformation | Callable]): functions to be called based on index.
        """
        super().__init__(fns)

        self.t_kwargs = t_kwargs

    def _t(self, index, *args, **kwargs):
        def _wrap_fn(i, args):
            _args, _kwargs = args
            _r, _kwargs = self.fn[i](*_args, **_kwargs)

            return _r, _kwargs

        _r, _kwargs = jax.lax.switch(
            index,
            tuple(partial(_wrap_fn, i) for i in range(len(self.fn))),
            (args, kwargs),
            **self.t_kwargs,
        )
        return _r, _kwargs
