__all__ = [
    "cond",
    "switch",
    "scan",
]


import jax
from typing import Union, Callable, Tuple

from ..core.transform import _AbstractTransformation
from ..core.modules import ParamDict
from ..core.filter import f


class cond(_AbstractTransformation):
    def __init__(
        self,
        true_fn: Union[_AbstractTransformation, Callable],
        false_fn: Union[_AbstractTransformation, Callable],
        filter: Union[f, Callable[[ParamDict], ParamDict]] = lambda *args: True,
    ):
        super().__init__((true_fn, false_fn), filter)

    def _call(self, params_partition, *args):
        output, params_partition = self.transform(
            params_partition,
            *args
        )

        return output

    def _make_transform(self):
        return lambda partition, cond, *args: jax.lax.cond(
            cond,
            *tuple(self._functional(fn) for fn in self.fn),
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
        output, params_partition = self.transform(
            params_partition,
            *args
        )

        return output

    def _make_transform(self):
        return lambda partition, j, *args: jax.lax.switch(
            j,
            tuple(self._functional(fn) for fn in self.fn),
            partition,
            *args,
        )


class scan(_AbstractTransformation):
    def __init__(
        self,
        fn: Union[_AbstractTransformation, Callable],
        filter: Union[f, Callable[[ParamDict], ParamDict]] = lambda *args: True,
        map_outputs: Tuple[int, ...] = (),
    ):
        super().__init__(fn, filter)
        self.map_outputs = map_outputs
        self.js = None
        self.length = None

    def _call(self, params_partition, *args):
        # Assert that exactly one between 'js' and 'length' is specified in self.kwargs
        assert sum(map(lambda x: x in self.kwargs, ("js", "length"))) == 1, \
            "Exactly one between 'js' and 'length' must be specified in kwargs"

        if "js" in self.kwargs:
            self.js = self.kwargs["js"]
            del self.kwargs["js"]

        if "length" in self.kwargs:
            self.length = self.kwargs["length"]
            del self.kwargs["length"]

        if self.js is not None:
            args = (None,) + args

        output, params_partition = self.transform(
            params_partition,
            *args
        )

        return output

    def _make_transform(self):
        def scan(
            carry, j
        ):
            partition, args_list = carry

            if self.js is not None:
                r, partition = self._functional(self.fn)(partition, j, *args_list[1:])
            else:
                r, partition = self._functional(self.fn)(partition, *args_list)

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
                    self.map_outputs + tuple(range(len(updated_args) - len(self.map_outputs)))
                ):
                    args_list[map_output] = updated_arg
            else:
                y = r

            return (partition, args_list), y

        return lambda partition, *args: jax.lax.scan(scan, (partition, args), self.js, self.length)
