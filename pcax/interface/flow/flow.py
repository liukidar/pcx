import functools
import jax
from ...utils.functions import all_kwargs, call_kwargs


def switch(index_fn, fns, **static_kwargs):
    @functools.wraps(fns[0])
    def wrapper(j, *args, **kwargs):
        return jax.lax.switch(
            index_fn(j),
            tuple(
                lambda kwargs, *args: fn(*args, **{**static_kwargs, **kwargs})
                for fn in fns
            ),
            kwargs,
            *args
        )

    return wrapper


def scan(fn, js=None, length=None, return_key="y", **static_kwargs):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        fn_kwargs = all_kwargs(fn, *args, **{**static_kwargs, **kwargs})

        for key in static_kwargs:
            del fn_kwargs[key]

        def body(fn_kwargs, j):
            if js is None:
                r_kwargs = call_kwargs(fn, **{**static_kwargs, **fn_kwargs})
            else:
                r_kwargs = call_kwargs(fn, j, **{**static_kwargs, **fn_kwargs})

            if return_key in r_kwargs:
                y = r_kwargs[return_key]
                del r_kwargs[return_key]
            else:
                y = None

            fn_kwargs.update(r_kwargs)

            return fn_kwargs, y

        return jax.lax.scan(body, fn_kwargs, js, length)

    return wrapper
