import functools
import inspect
from logging import warning
import jax
import equinox as eqx
from .. import _C


def switch(index_fn, fns, **static_kwargs):
    @functools.wraps(fns[0])
    def wrapper(j, *args, **kwargs):
        return jax.lax.switch(
            index_fn(j),
            tuple(
                lambda args, kwargs: fn(*args, **kwargs, **static_kwargs) for fn in fns
            ),
            args,
            kwargs,
        )

    return wrapper


def scan(fn, js=None, length=None, **static_kwargs):
    parameter_names = inspect.signature(fn).parameters

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Convert args to kwargs
        kwargs.update(zip(parameter_names, args))

        if _C["debug"]:
            (_, static_fields) = eqx.partition(kwargs, lambda _: True)
            debug_hash = hash(tuple(static_fields.items()))

        def body(kwargs, j):
            if j is None:
                r = fn(**kwargs, **static_kwargs)
            else:
                r = fn(j, **kwargs, **static_kwargs)

            # Update args
            if isinstance(r[0], dict):
                kwargs.update(r[0])
            elif isinstance(r[0], tuple | list):
                # If the return value is an iterable, its elements will replace fn's arguments in order.
                kwargs.update(zip(parameter_names, r[0]))
            else:
                raise TypeError(
                    f"Invalid type for the first element of the return value: {type(r[0])}."
                    "A scanned function must return a tuple of length 2 or less, "
                    "where the first element is a dictionary|tuple|list of argument updates."
                )

            if len(r) == 2:
                y = r[1]
            else:
                y = None

            return kwargs, y

        r, y = jax.lax.scan(body, kwargs, js, length)

        if _C["debug"]:
            (_, static_fields) = eqx.partition(r, lambda _: True)
            if debug_hash != hash(tuple(static_fields.items())):
                warning(
                    f"Function '{fn.__name__}' has modified its static arguments"
                    " within a scan loop. Remember that the changes are not propagated"
                    " to the next iteration as the function is compiled before being executed."
                    " Changes to static arguments take effect only after the scan loop has finished."
                    " If this behaviour is not desired, consider using a for loop instead (as it will be unrolled)."
                    " If this is instead intended, ignore this warning."
                )

        return r, y

    return wrapper
