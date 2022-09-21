import jax

jax.config.update("jax_platform_name", "cpu")


def f(a):
    print("Compile f")
    return jax.lax.cond(
        a is not None,
        lambda b: 0 if b is None else b,
        lambda b: 0,
        a,
    )


def g(a):
    return 0


a = 1


@jax.jit
def run(a):
    print("Compiling")

    def loop(i, a_r):
        a, r = a_r
        r += jax.lax.cond(a[0] is None, g, f, a[0])
        return (None, 1), r

    r = jax.lax.fori_loop(0, 2, loop, (a, 0))
    return r[1]


# run((1, 1))
print(run((1, 1)))
