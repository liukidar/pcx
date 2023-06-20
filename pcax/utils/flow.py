import functools
import jax
from typing import Union, Callable, Optional, Tuple, Any

from ..core.util import make_args, kwargs_indices
from ..core.transform import ModuleTransform
from ..core.structure import Module, Function, VarCollection
from ..core.filter import _
from ..core.random import RKG


class Switch(ModuleTransform):
    def __init__(
        self,
        fns: Tuple[Union[Module, Callable], ...],
        vc_f: Union[_, Callable[[VarCollection], VarCollection]] = lambda vc: vc,
        static_argnums: Tuple[int, ...] = (),
    ):
        super().__init__(Function(fns[0], sum(map(lambda fn: fn.vars(), fns), VarCollection())), vc_f)

        def switch(f):
            @self._functional
            def switched(params, args_list):
                self.vc_target.load(*params)

                for static_argnum in self.static_argnums:
                    args_list[static_argnum] = self.static_args[static_argnum]

                return f(*args_list), self.vc_target.dump()[:2]

            return switched

        self._call = lambda j, params, args_list: jax.lax.switch(
            j,
            self.fns,
            params,
            args_list,
        )
        self.fns = tuple(switch(f) for f in fns)
        self.static_argnums = static_argnums
        self.static_args = {}

    @property
    def vc_target(self) -> VarCollection:
        return self.vc_f(self.vc) + VarCollection(RKG.vars())

    def __call__(self, j: int, *args, **kwargs):
        differentiable, dynamic, static = self.vc_target.dump()

        args_list = make_args(
            self.__wrapped__,
            args,
            kwargs
        )

        for static_argnum in self.static_argnums:
            self.static_args[static_argnum] = args_list[static_argnum]
            args_list[static_argnum] = None

        output, changes = self._call(
            j, (differentiable, dynamic), args_list
        )
        self.vc_target.load(*changes)
        self.static_args = {}

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}(f={self.__wrapped__})"


class Scan(ModuleTransform):
    def __init__(
        self,
        f: Union[Module, Callable],
        vc_f: Union[_, Callable[[VarCollection], VarCollection]] = lambda vc: vc,
        js: Optional[Tuple[Any, ...]] = None,
        length: Optional[int] = None,
        map_outputs: Tuple[int, ...] = (),
        static_argnums: Tuple[int, ...] = (),
    ):
        super().__init__(f, vc_f)

        # not _functional to catch if the user changes static values inside the vc
        def scan(
            carry, j
        ):
            params, args_list = carry
            self.vc_target.load(*params)

            for static_argnum in self.static_argnums:
                args_list[static_argnum] = self.static_args[static_argnum]

            if self.has_aux:
                r = f(j, *args_list[1:])
            else:
                r = f(*args_list)

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
                    map_outputs + tuple(range(len(updated_args) - len(map_outputs)))
                ):
                    args_list[map_output] = updated_arg
            else:
                y = r

            # return only the target because it's the value passed in input,
            # other changes are not captured not permitted since the shape must be constant.
            return (self.vc_target.dump()[:2], args_list), y

        self._call = lambda params, args_list: jax.lax.scan(scan, (params, args_list), js, length)
        self.static_argnums = static_argnums
        self.static_args = {}
        self.has_aux = js is not None

    @property
    def vc_target(self) -> VarCollection:
        return self.vc_f(self.vc) + VarCollection(RKG.vars())

    def __call__(self, *args, **kwargs):
        differentiable, dynamic, static = self.vc_target.dump()

        if self.has_aux:
            args = (None,) + args

        args_list = make_args(
            self.__wrapped__,
            args,
            kwargs
        )

        for static_argnum in self.static_argnums:
            self.static_args[static_argnum] = args_list[static_argnum]
            args_list[static_argnum] = None

        (changes, _), output = self._call(
            (differentiable, dynamic), args_list
        )
        self.vc_target.load(*changes)
        self.static_args = {}

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}(f={self.__wrapped__})"


def scan(f):
    @functools.wraps(f)
    def wrapper(
        filter: Union[_, Callable[[VarCollection], VarCollection]] = lambda vc: vc,
        js: Optional[Tuple[Any, ...]] = None,
        length: Optional[int] = None,
        map_outputs: Tuple[int, ...] = (),
        **static_kwargs
    ):
        scan = Scan(
            f,
            vc_f=filter,
            js=js,
            length=length,
            map_outputs=map_outputs,
            static_argnums=kwargs_indices(f, static_kwargs),
        )

        return Function(
            functools.wraps(f)(lambda *args, **kwargs: scan(*args, **{**dict(static_kwargs), **kwargs})),
            scan.vc
        )

    return wrapper


def switch(*fns):
    def wrapper(
        filter: Union[_, Callable[[VarCollection], VarCollection]] = lambda vc: vc,
        **static_kwargs
    ):
        switch = Switch(
            fns,
            vc_f=filter,
            static_argnums=kwargs_indices(fns[0], static_kwargs),
        )

        return Function(
            lambda j, *args, **kwargs: switch(j, *args, **{**dict(static_kwargs), **kwargs}),
            switch.vc
        )

    return wrapper
