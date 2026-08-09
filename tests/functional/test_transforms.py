"""Equivalence contracts for `pxf.jit`, `pxf.value_and_grad` and `pxf.vmap`.

Every pcx transform is a thin wrapper around a jax primitive, which hands us a
free and completely independent oracle: the pcx version must compute exactly what
the equivalent hand-written raw-jax version computes. Every expectation in this
file is therefore written twice — once through pcx, once through `jax.jit` /
`jax.value_and_grad` / `jax.vmap` on a plain function over plain arrays — and the
two are compared. No expected value is ever taken from what pcx happens to
return.

On top of numerical agreement, the transforms carry a protocol of their own
(documented in the header of `pcx/functional/_transform.py`):

* positional arguments are pure jax values — mutations to them are discarded;
* keyword arguments are *tracked* — a `Param` mutated inside is written back into
  the caller's own object;
* the same `Param` reachable twice in the kwargs is one parameter, so it is
  written back once and its gradient is the sum over both paths;
* a reserved `__RKG` kwarg carries the global `pcx.RKG` through the transform.

Those four are what make pcx usable at all, and none of them is checked by jax.
"""

import jax
import jax.numpy as jnp
import pytest
from conftest import assert_allclose

import pcx
import pcx.functional as pxf
import pcx.utils as pxu


def diff_params():
    """Mask selecting every `px.Param` in the kwargs for differentiation.

    `M(...).to((False, True))` maps unmatched leaves to False and matched ones to
    True, which is the True/False-per-parameter mask `ValueAndGrad` documents.
    A fresh mask is built per call because `.to()` mutates the mask in place.
    """
    return pxu.M(pcx.Param).to((False, True))


# Reference computations. These are plain functions of plain arrays, so they can
# be handed to raw jax unchanged; the pcx-side tests express the same maths
# through Params.


def _affine(x, w, b):
    return jnp.tanh(x * w + b)


def _affine_loss(w, x):
    return jnp.sum(_affine(x, w, 1.0))


# jit ##################################################################################################################


def test_jit_matches_raw_jax_on_the_same_computation():
    """`pxf.jit` is documented to "behave exactly as its jax counterpart".

    If the wrapper perturbed the computation — reordering arguments, dropping a
    term, quietly casting — every downstream number would be wrong while the
    program still ran, so this is the first thing that has to hold.
    """
    x = jnp.array([0.5, -1.5])
    w = jnp.array([2.0, 3.0])
    b = jnp.array([-0.25, 0.75])

    assert_allclose(pxf.jit()(_affine)(x, w, b), jax.jit(_affine)(x, w, b))


def test_jit_matches_raw_jax_across_repeated_calls_with_new_values():
    """A jitted function is compiled once and replayed; a wrapper that captured a
    traced value in a Python closure would return the *first* call's answer
    forever. Two calls with different inputs pin that down."""
    jitted = pxf.jit()(_affine)

    for x in (jnp.array([0.0, 1.0]), jnp.array([2.0, -3.0]), jnp.array([0.25, 0.25])):
        w, b = jnp.array([1.5, -0.5]), jnp.array([0.125, 2.0])
        assert_allclose(jitted(x, w, b), _affine(x, w, b))


def test_jit_matches_raw_jax_with_static_argnums():
    """`static_argnums` is forwarded to `jax.jit`. A Python `int` that arrives as
    a tracer instead of a constant cannot drive a `range`, so this checks the
    argument really is being held static."""

    @pxf.jit(static_argnums=0)
    def repeat(n, x):
        for _ in range(n):
            x = x * 2.0
        return x

    def reference(n, x):
        for _ in range(n):
            x = x * 2.0
        return x

    x = jnp.array([1.0, -2.0])
    assert_allclose(repeat(3, x), jax.jit(reference, static_argnums=0)(3, x))


def test_jit_writes_back_a_param_passed_as_a_keyword():
    """The headline feature: keyword arguments are tracked, so a `Param` mutated
    inside the jit barrier is updated in the caller's own object.

    Without this every stateful layer — batch norm statistics, Vode states,
    optimiser slots — would silently keep its pre-call value and training would
    quietly do nothing.
    """
    start = jnp.array([1.0, 2.0])
    x = jnp.array([1.0, 1.0])
    param = pcx.Param(start)

    @pxf.jit()
    def step(x, *, p):
        p.set(p.get() * 2.0 + x)
        return p.get().sum()

    returned = step(x, p=param)

    expected = jax.jit(lambda v, x: v * 2.0 + x)(start, x)
    assert_allclose(param.get(), expected)
    assert_allclose(returned, expected.sum())


def test_jit_does_not_write_back_a_param_passed_positionally():
    """The mirror-image half of the protocol: positional arguments are "pure jax
    ones (i.e., not tracked)".

    A user who passes a model positionally must see it unchanged; if positional
    mutations leaked back, the two argument kinds would be indistinguishable and
    the documented way to pass a read-only model would not exist.
    """
    start = jnp.array([1.0, 2.0])
    x = jnp.array([1.0, 1.0])
    param = pcx.Param(start)

    @pxf.jit()
    def step(p, x):
        p.set(p.get() * 2.0 + x)
        return p.get().sum()

    returned = step(param, x)

    # The value returned still reflects the mutation inside the transform...
    assert_allclose(returned, jax.jit(lambda v, x: (v * 2.0 + x).sum())(start, x))
    # ...but the caller's object is untouched.
    assert_allclose(param.get(), start)


def test_jit_updates_a_shared_param_exactly_once():
    """Parameter sharing (NOTE #2 of the transform protocol): the same `Param`
    reachable under two kwargs is *one* parameter.

    If the write-back applied per reference instead of per object, a tied weight
    would be updated twice — here `+1.0` would become `+2.0` — and every tied
    architecture would train at a silently doubled rate.
    """
    param = pcx.Param(jnp.array([1.0]))
    seen = {}

    @pxf.jit()
    def bump(*, a, b):
        seen["same_object"] = a is b
        a.set(a.get() + 1.0)
        return b.get()

    returned = bump(a=param, b=param)

    assert seen["same_object"] is True, "a shared Param must arrive as one object, not two copies"
    assert_allclose(param.get(), jnp.array([2.0]))
    assert_allclose(returned, jnp.array([2.0]))


def test_jit_writes_back_a_param_shared_between_a_module_and_a_kwarg():
    """The realistic form of sharing: a tied weight reachable both directly and
    through a module. The update must land once, in the one object both paths
    point at."""

    class Wrapper(pcx.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p

    param = pcx.Param(jnp.array([3.0]))
    wrapper = Wrapper(param)

    @pxf.jit()
    def bump(*, model, p):
        model.p.set(model.p.get() + 1.0)
        return p.get()

    returned = bump(model=wrapper, p=param)

    assert wrapper.p is param
    assert_allclose(param.get(), jnp.array([4.0]))
    assert_allclose(returned, jnp.array([4.0]))


# value_and_grad #######################################################################################################


def test_value_and_grad_matches_jax_value_and_grad():
    """The value *and* the gradient must both agree with `jax.value_and_grad` on
    the equivalent pure function. A wrapper that differentiated a rearranged or
    partially-masked version of the function would still return plausible
    numbers, so both halves are compared."""
    w = jnp.array([0.5, -1.25])
    x = jnp.array([2.0, 3.0])
    param = pcx.Param(w)

    @pxf.value_and_grad(diff_params())
    def loss(x, *, p):
        return _affine_loss(p.get(), x)

    value, grads = loss(x, p=param)
    expected_value, expected_grad = jax.value_and_grad(_affine_loss)(w, x)

    assert_allclose(value, expected_value)
    assert_allclose(grads["p"].get(), expected_grad)


def test_value_and_grad_leaves_undifferentiated_params_out_of_the_gradient():
    """The mask decides what is differentiated. A parameter the mask excludes must
    come back as `None`, not as a zero array and certainly not as a gradient: the
    optimiser walks this tree, and a spurious leaf would make it step a frozen
    weight."""
    w = jnp.array([0.5, -1.25])
    x = jnp.array([2.0, 3.0])
    trainable = pcx.Param(w)
    frozen = pcx.Param(jnp.array([9.0, 9.0]))

    # Differentiate only the parameter that is *not* marked frozen.
    mask = pxu.M_hasnot(pcx.Param, frozen=True).to((False, True))
    frozen.frozen = True

    @pxf.value_and_grad(mask)
    def loss(x, *, p, q):
        return _affine_loss(p.get(), x) + 0.0 * q.get().sum()

    _, grads = loss(x, p=trainable, q=frozen)

    assert grads["q"] is None, "a masked-out parameter must not receive a gradient"
    assert_allclose(grads["p"].get(), jax.grad(_affine_loss)(w, x))


def test_value_and_grad_matches_jax_when_the_function_has_auxiliary_outputs():
    """With `has_aux=True` the extra returns must survive untouched. pcx packs
    everything after the differentiated scalar into a tuple, so a function may
    return several extras; their *values* still have to be exactly jax's."""
    w = jnp.array([0.5, -1.25])
    x = jnp.array([2.0, 3.0])
    param = pcx.Param(w)

    def reference(w, x):
        y = _affine(x, w, 1.0)
        return y.sum(), (y, w * 2.0)

    @pxf.value_and_grad(diff_params(), has_aux=True)
    def loss(x, *, p):
        y = _affine(x, p.get(), 1.0)
        return y.sum(), y, p.get() * 2.0

    (value, aux), grads = loss(x, p=param)
    (expected_value, expected_aux), expected_grad = jax.value_and_grad(reference, has_aux=True)(w, x)

    assert_allclose(value, expected_value)
    assert_allclose(grads["p"].get(), expected_grad)
    assert len(aux) == 2, "both extra returns must be handed back"
    assert_allclose(aux[0], expected_aux[0])
    assert_allclose(aux[1], expected_aux[1])


def test_value_and_grad_writes_back_a_param_mutated_in_the_function():
    """Differentiating must not switch off tracking: a stateful side effect inside
    the loss (a batch-norm running mean, a Vode cache) still has to reach the
    caller's object, exactly as under `jit`."""
    w = jnp.array([0.5, -1.25])
    x = jnp.array([2.0, 3.0])
    param = pcx.Param(w)
    counter = pcx.Param(jnp.array([0.0]))

    @pxf.value_and_grad(diff_params())
    def loss(x, *, p, c):
        c.set(c.get() + 1.0)
        return _affine_loss(p.get(), x)

    value, grads = loss(x, p=param, c=counter)

    assert_allclose(value, _affine_loss(w, x))
    assert_allclose(grads["p"].get(), jax.grad(_affine_loss)(w, x))
    assert_allclose(counter.get(), jnp.array([1.0]))
    # The differentiated parameter itself is never modified by the transform.
    assert_allclose(param.get(), w)


@pytest.mark.bug(
    "BUGS.md#6: ValueAndGrad hands positional args straight to the function, so mutating one writes into the caller"
)
def test_value_and_grad_does_not_write_back_a_param_passed_positionally():
    """Positional arguments must stay pure under `value_and_grad` exactly as they
    do under `jit` — the protocol is a property of the library, not of one
    transform.

    `Jit` flattens its arguments through `jax.jit`, so the function inside gets a
    reconstructed `Param` and the caller's object survives untouched. `ValueAndGrad`
    forwards `*args` straight into `jax.value_and_grad`, which leaves
    non-differentiated positional arguments as the very objects passed in, so
    `c.set(...)` mutates the caller's `Param` in place. Identical code therefore
    behaves differently depending on which transform wraps it.
    """
    w = jnp.array([0.5, -1.25])
    x = jnp.array([2.0, 3.0])
    positional = pcx.Param(jnp.array([0.0]))
    param = pcx.Param(w)

    @pxf.value_and_grad(diff_params())
    def loss(c, x, *, p):
        c.set(c.get() + 1.0)
        return _affine_loss(p.get(), x)

    loss(positional, x, p=param)

    assert_allclose(positional.get(), jnp.array([0.0]))


@pytest.mark.bug(
    "BUGS.md#6: a positional Param mutated inside value_and_grad keeps a live autodiff tracer after the call returns"
)
def test_value_and_grad_does_not_leak_a_tracer_into_a_positional_param():
    """The severe form of the same defect. When the mutation involves the
    differentiated parameter, the value written into the caller's positional
    `Param` is an autodiff tracer rather than an array.

    Nothing raises at the time; the object is only detonated later, wherever it is
    next read, as an `UnexpectedTracerError` with no connection to the call that
    caused it.
    """
    w = jnp.array([0.5, -1.25])
    x = jnp.array([2.0, 3.0])
    positional = pcx.Param(jnp.array([0.0]))
    param = pcx.Param(w)

    @pxf.value_and_grad(diff_params())
    def loss(c, x, *, p):
        c.set(c.get() + p.get().sum())
        return _affine_loss(p.get(), x)

    loss(positional, x, p=param)

    leaked = type(positional.get()).__name__
    assert "Tracer" not in leaked, f"the caller's positional Param now holds a {leaked}"


def test_value_and_grad_sums_the_gradient_of_a_shared_param_over_both_paths():
    """A tied weight used twice must receive the *sum* of the two gradients — that
    is what differentiating the equivalent single-variable function gives.

    Treating the two references as independent leaves would halve the effective
    learning signal of every tied weight, and nothing would raise.
    """
    v = jnp.array([1.5, -2.0])
    x = jnp.array([2.0, 3.0])
    param = pcx.Param(v)

    def reference(v, x):
        return jnp.sum(v * x) + jnp.sum(v**2)

    @pxf.value_and_grad(diff_params())
    def loss(x, *, a, b):
        return jnp.sum(a.get() * x) + jnp.sum(b.get() ** 2)

    value, grads = loss(x, a=param, b=param)
    expected_value, expected_grad = jax.value_and_grad(reference)(v, x)

    assert_allclose(value, expected_value)
    # d/dv [v*x + v^2] = x + 2v, i.e. the sum over both paths.
    assert_allclose(grads["a"].get(), expected_grad)
    assert grads["b"] is None, "the second reference is an alias, not a second gradient"


def test_value_and_grad_updates_a_shared_param_exactly_once():
    """As under `jit`: one object, one write-back, however many kwargs point at
    it."""
    param = pcx.Param(jnp.array([1.0]))
    state = pcx.Param(jnp.array([0.0]))

    @pxf.value_and_grad(diff_params())
    def loss(*, p, a, b):
        a.set(a.get() + 1.0)
        return p.get().sum()

    loss(p=param, a=state, b=state)

    assert_allclose(state.get(), jnp.array([1.0]))


@pytest.mark.bug(
    "#74: pxf.value_and_grad(argnums=0) raises TypeError: an int argnums is splatted with * instead of wrapped"
)
def test_value_and_grad_accepts_an_integer_argnums():
    """`pcx.functional.value_and_grad` advertises `argnums: int | Sequence[int]`,
    matching `jax.value_and_grad`, and a bare `int` is by far the common way to
    ask for the gradient of a single positional argument.

    `ValueAndGrad._t` builds `(*self.t_kwargs.get("argnums", ()), len(args))`,
    which requires `argnums` to be iterable, so the documented `int` form is
    rejected outright. Only the undocumented `(0,)` spelling works.
    """
    w = jnp.array([0.5, -1.25])
    x = jnp.array([2.0, 3.0])
    param = pcx.Param(w)

    @pxf.value_and_grad(diff_params(), argnums=0)
    def loss(x, *, p):
        return _affine_loss(p.get(), x)

    value, grads = loss(x, p=param)
    expected_value, expected_dx = jax.value_and_grad(lambda x, w: _affine_loss(w, x))(x, w)

    assert_allclose(value, expected_value)
    positional_grads, kwarg_grads = grads
    leaves = jax.tree_util.tree_leaves(positional_grads)
    assert len(leaves) == 1, "argnums=0 asks for exactly one positional gradient"
    assert_allclose(leaves[0], expected_dx)
    assert_allclose(kwarg_grads["p"].get(), jax.grad(_affine_loss)(w, x))


# Composition ##########################################################################################################


def test_jit_of_value_and_grad_matches_the_unnested_form():
    """Transforms are documented as composable (`_BaseTransform` accepts another
    transform as its `fn`). Composition must be purely an optimisation: jitting a
    gradient computation may not change a single number."""
    w = jnp.array([0.5, -1.25])
    x = jnp.array([2.0, 3.0])

    def build():
        return pxf.value_and_grad(diff_params())(lambda x, *, p: _affine_loss(p.get(), x))

    plain_param = pcx.Param(w)
    plain_value, plain_grads = build()(x, p=plain_param)

    jitted_param = pcx.Param(w)
    jitted_value, jitted_grads = pxf.jit()(build())(x, p=jitted_param)

    expected_value, expected_grad = jax.value_and_grad(_affine_loss)(w, x)

    assert_allclose(plain_value, expected_value)
    assert_allclose(jitted_value, expected_value)
    assert_allclose(plain_grads["p"].get(), expected_grad)
    assert_allclose(jitted_grads["p"].get(), expected_grad)


def test_jit_of_value_and_grad_still_writes_back_through_both_transforms():
    """Nesting must not drop the tracking protocol on the way through: the inner
    transform hands its kwargs out to the outer one, and only the outermost call
    injects them back into the caller's objects."""
    w = jnp.array([0.5, -1.25])
    x = jnp.array([2.0, 3.0])
    param = pcx.Param(w)
    counter = pcx.Param(jnp.array([0.0]))

    inner = pxf.value_and_grad(diff_params())(lambda x, *, p, c: (c.set(c.get() + 1.0), _affine_loss(p.get(), x))[1])
    _, grads = pxf.jit()(inner)(x, p=param, c=counter)

    assert_allclose(grads["p"].get(), jax.grad(_affine_loss)(w, x))
    assert_allclose(counter.get(), jnp.array([1.0]))


# vmap #################################################################################################################
#
# Every test below is marked `bug`: `Vmap._t` calls `_process_mask` without an
# `rkg_mask`, so the injected `mask["__RKG"]` is `None` while the kwargs actually
# hold a `RandomKeyGenerator`, and `jtu.tree_map` rejects the mismatch before the
# transform ever runs. The assertions state what vmap is supposed to do.


def test_vmap_matches_jax_vmap():
    """The baseline equivalence: mapping a function over a leading axis must give
    exactly what `jax.vmap` gives."""

    @pxf.vmap(in_axes=(0,), out_axes=0)
    def scaled(row):
        return jnp.tanh(row * 2.0)

    xs = jnp.arange(6.0).reshape(3, 2)

    assert_allclose(scaled(xs), jax.vmap(lambda row: jnp.tanh(row * 2.0))(xs))


def test_vmap_broadcasts_unmapped_positional_arguments_like_jax():
    """`in_axes=None` marks an argument as shared across the mapped axis; getting
    this wrong would silently map over the wrong operand."""

    @pxf.vmap(in_axes=(0, None), out_axes=0)
    def shifted(row, offset):
        return row + offset

    xs = jnp.arange(6.0).reshape(3, 2)
    offset = jnp.array([10.0, 100.0])

    assert_allclose(shifted(xs, offset), jax.vmap(lambda r, o: r + o, in_axes=(0, None))(xs, offset))


def test_vmap_writes_back_a_mapped_param():
    """A mapped `Param` is the pcx idiom for per-example state (a Vode's `h`).
    The per-example updates must come back stacked into the caller's object."""
    start = jnp.arange(3.0)
    param = pcx.Param(start)

    @pxf.vmap(pxu.M(pcx.Param).to((None, 0)), in_axes=(0,), out_axes=0)
    def accumulate(x, *, p):
        p.set(p.get() + x)
        return p.get()

    xs = jnp.ones(3)
    returned = accumulate(xs, p=param)

    assert_allclose(returned, start + xs)
    assert_allclose(param.get(), start + xs)


def test_vmap_does_not_write_back_a_param_passed_positionally():
    """Positional arguments stay pure under `vmap` too."""
    start = jnp.arange(3.0)
    param = pcx.Param(start)

    @pxf.vmap(in_axes=(0, 0), out_axes=0)
    def accumulate(x, p):
        p.set(p.get() + x)
        return p.get()

    returned = accumulate(jnp.ones(3), param)

    assert_allclose(returned, start + 1.0)
    assert_allclose(param.get(), start)


def test_vmap_updates_a_shared_param_exactly_once():
    """One object, one write-back — the sharing rule is independent of which
    transform is applied."""
    start = jnp.arange(3.0)
    param = pcx.Param(start)

    @pxf.vmap(pxu.M(pcx.Param).to((None, 0)), in_axes=(0,), out_axes=0)
    def bump(x, *, a, b):
        a.set(a.get() + x)
        return b.get()

    returned = bump(jnp.ones(3), a=param, b=param)

    assert_allclose(returned, start + 1.0)
    assert_allclose(param.get(), start + 1.0)


# Randomness ###########################################################################################################
#
# These come last: the exception test can leave a tracer installed in the global
# RKG, and although conftest's autouse fixture reseeds it, ordering keeps any
# fallout away from the tests above.


def test_jit_advances_the_global_rkg_exactly_like_a_raw_jax_split():
    """`RKG` is injected into every transform as the reserved `__RKG` kwarg, so a
    draw inside the jit barrier must (a) produce the same sample raw jax would and
    (b) leave the *caller's* global key advanced.

    `RKGState.split(1)` is `jax.random.split(key, 2)`, keeping `[0]` as the new
    state and handing out `[1]`. If the new state failed to escape the barrier,
    every jitted call would draw the same numbers forever.
    """
    pcx.RKG.seed(0)

    @pxf.jit()
    def draw(x):
        return x + jax.random.normal(pcx.RKG(), (2,))

    sample = draw(jnp.zeros(2))

    fresh = jax.random.split(jax.random.PRNGKey(0), 2)
    assert_allclose(sample, jax.random.normal(fresh[1], (2,)))
    assert jnp.array_equal(pcx.RKG.key.get(), fresh[0]), "the global key must be advanced past the consumed one"


def test_repeated_jitted_draws_follow_the_raw_jax_key_stream():
    """The state has to keep advancing on *every* call, not just the tracing one.
    A cached executable that reused the key captured at trace time would return
    identical "random" batches for the rest of the run."""
    pcx.RKG.seed(0)

    @pxf.jit()
    def draw(x):
        return x + jax.random.normal(pcx.RKG(), (2,))

    key = jax.random.PRNGKey(0)
    for call in range(3):
        sample = draw(jnp.zeros(2))
        fresh = jax.random.split(key, 2)
        assert_allclose(sample, jax.random.normal(fresh[1], (2,)), err_msg=f"call {call}")
        key = fresh[0]
        assert jnp.array_equal(pcx.RKG.key.get(), key), f"global key diverged at call {call}"


def test_global_rkg_holds_a_concrete_array_after_a_transformed_function_raises():
    """`_BaseTransform` swaps `RKG.key` for the traced key on the way in and
    restores it on the way out — but the restore is a plain statement, not a
    `finally`. If the wrapped function raises, the traced key stays installed in
    module-level state.

    The damage is not local: `pcx.RKG` is the default argument of every layer and
    Vode constructor, so from that point on the *whole process* builds models from
    a dead tracer and fails with an unrelated `UnexpectedTracerError` far from the
    real cause. A failed training step must not poison the interpreter.

    Kept last in the file, since a failure here leaks global state.
    """
    pcx.RKG.seed(0)

    @pxf.jit()
    def boom(x):
        raise ValueError("deliberate failure inside the transform")

    with pytest.raises(ValueError, match="deliberate failure"):
        boom(jnp.zeros(2))

    key_type = type(pcx.RKG.key.get()).__name__
    assert "Tracer" not in key_type, f"pcx.RKG.key holds a {key_type} after the failed call, not a concrete array"
