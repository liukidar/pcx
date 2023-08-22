__all__ = [
    "cond",
    "switch",
    "scan",
]


from typing import Any, Callable, Optional, Tuple, Union

import jax

from ..core.filter import f
from ..core.modules import ParamDict
from ..core.transform import _AbstractTransformation


class cond(_AbstractTransformation):
    def __init__(
        self,
        true_fn: Union[_AbstractTransformation, Callable],
        false_fn: Union[_AbstractTransformation, Callable],
        filter: Union[f, Callable[[ParamDict], ParamDict]] = lambda *args: True,
    ):
        super().__init__((true_fn, false_fn), filter)

    def _call(self, params_partition, *args):
        output, params_partition = self.transform(params_partition, *args)

        return output

    def _make_transform(self, fns, kwargs):
        return lambda partition, cond, *args: jax.lax.cond(
            cond,
            *tuple(self._functional(fn, kwargs) for fn in fns),
            partition,
            *args,
        )


class switch(_AbstractTransformation):
    def __init__(
        self,
        fns: Tuple[Union[_AbstractTransformation, Callable], ...],
        filter: Union[f, Callable[[ParamDict], ParamDict]] = lambda *args: True,
    ):
        super().__init__(fns, filter)

    def _call(self, params_partition, *args):
        output, params_partition = self.transform(params_partition, *args)

        return output

    def _make_transform(self, fns, kwargs):
        return lambda partition, j, *args: jax.lax.switch(
            j,
            tuple(self._functional(fn, kwargs) for fn in fns),
            partition,
            *args,
        )


class scan(_AbstractTransformation):
    def __init__(
        self,
        fn: Union[_AbstractTransformation, Callable],
        js: Optional[Union[jax.Array, Any]] = None,
        length: Optional[int] = None,
        map_outputs: Tuple[int, ...] = (),
        filter: Union[f, Callable[[ParamDict], ParamDict]] = lambda *args: True,
    ):
        assert (
            sum((js is not None, length is not None)) == 1
        ), "Exactly one between 'js' and 'length' must be specified"

        super().__init__(fn, filter)

        self.js = js
        self.length = length
        self.map_outputs = map_outputs

    def _call(self, params_partition, *args):
        if self.js is not None:
            args = (None,) + args

        (params_partition, args), output = self.transform(params_partition, *args)

        return output, *args

    def _make_transform(self, fn, kwargs):
        def scan(carry, j):
            partition, args_list = carry

            if self.js is not None:
                r, partition = self._functional(fn, kwargs)(
                    partition, j, *args_list[1:]
                )
            else:
                r, partition = self._functional(fn, kwargs)(partition, *args_list)

            # Update args
            if isinstance(r, tuple):
                if len(r) == 2 and isinstance(r[0], tuple):
                    updated_args = r[0]
                    y = r[1]
                else:
                    updated_args = r
                    y = None

                updated_args = r[0]
                for updated_arg, map_output in zip(
                    updated_args,
                    self.map_outputs
                    + tuple(range(len(updated_args) - len(self.map_outputs))),
                ):
                    args_list[map_output] = updated_arg
            else:
                y = r

            return (partition, args_list), y

        return lambda partition, *args: jax.lax.scan(
            scan, (partition, args), self.js, self.length
        )


class while_loop(_AbstractTransformation):
    def __init__(
        self,
        fn: Union[_AbstractTransformation, Callable],
        cond_fn: Callable,
        filter: Union[f, Callable[[ParamDict], ParamDict]] = lambda *args: True,
    ):
        """while_loop constructor.

        Args:
        fn: function corresponding to `body_fun` for jax.lax.while_loop,
        cond_fn: function corresponding to `cond_fun` for jax.lax.while_loop,
        filter: selects which params to apply the transformation to [
            it is used by vmap, grad, ... to select which params to be targeted by those transformations.
            There is no apparent use of it for flow transformations, but maybe I'm missing it;
            so there is still an option to specify it
        ],
        """
        super().__init__(fn, filter)

        self.cond_fn = cond_fn

    def _call(self, params_partition, *args):
        params_partition, output = self.transform(params_partition, *args)

        return output

    def _make_transform(self, fn, kwargs):
        def while_loop(carry):
            partition, args_list = carry
            updated_args, partition = self._functional(fn, kwargs)(
                partition, *args_list
            )
            assert len(updated_args) == len(args_list)

            return (partition, updated_args)

        return lambda partition, *args: jax.lax.while_loop(
            lambda carry: self.cond_fn(*carry[1]),
            while_loop,
            (partition, args),
        )
