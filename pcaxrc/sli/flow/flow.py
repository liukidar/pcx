import jax


def switch(i, fns, *args, **kwargs):
    return jax.lax.switch(
        i, tuple(lambda kwargs, *args: fn(*args, **kwargs) for fn in fns), kwargs, *args
    )
