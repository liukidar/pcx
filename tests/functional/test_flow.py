"""Equivalence contracts for the `jax.lax` control-flow wrappers.

`pxf.scan`, `pxf.while_loop`, `pxf.cond` and `pxf.switch` are described in
`pcx/functional/_flow.py` as "a thin wrapper around the corresponding jax.lax
function" that "behaves analogously to its jax counterpart", differing only in
how arguments are arranged. That gives every test below a free, independent
oracle: the same loop or branch written directly against `jax.lax`. Nothing here
takes its expected value from what pcx returns.

The calling conventions the wrappers add on top of `jax.lax`:

* `Scan`'s body is `fn(x, *args, **kwargs) -> (carry, y)` — the scanned element
  comes first, and the carry is returned as a tuple;
* `WhileLoop` takes `cond_fun` as a keyword and gives it the same `*args,
  **kwargs` as the body;
* `Cond` and `Switch` take the predicate/index as their first call argument.

And on top of that the shared pcx protocol: positional arguments are pure jax
values, keyword arguments are tracked so `Param` mutations propagate back to the
caller, the same `Param` referenced twice is one parameter, and a reserved
`__RKG` kwarg threads the global RNG through the primitive.
"""

import jax
import jax.numpy as jnp
import pytest
from conftest import assert_allclose

import pcx
import pcx.functional as pxf
import pcx.utils as pxu

# Reference computations: plain functions of plain arrays, usable by raw jax
# unchanged. The pcx side expresses the same maths through Params.

XS = jnp.array([1.0, 2.0, 3.0, 4.0])


def _scan_step(carry, x):
    """One step of a decaying accumulator; returns `(new_carry, per_step_output)`."""
    return carry * 0.5 + x, carry


# scan #################################################################################################################


def test_scan_matches_jax_lax_scan():
    """Both the final carry and the stacked per-step outputs must equal
    `jax.lax.scan`'s. A wrapper that dropped `y`, stacked it in the wrong order or
    lost a step would still return arrays of the right shape."""

    def body(x, carry):
        return _scan_step(carry, x)

    (carry,), ys = pxf.scan(body, XS)(jnp.array(0.0))
    expected_carry, expected_ys = jax.lax.scan(_scan_step, jnp.array(0.0), XS)

    assert_allclose(carry, expected_carry)
    assert_allclose(ys, expected_ys)


def test_scan_matches_jax_lax_scan_when_reversed():
    """`reverse=True` is forwarded to `jax.lax.scan`, and reversal affects both the
    order steps run in and the order outputs are stacked. Getting only one of the
    two right is a plausible failure that ordinary use would not surface."""

    def body(x, carry):
        return _scan_step(carry, x)

    (carry,), ys = pxf.scan(body, XS, reverse=True)(jnp.array(0.0))
    expected_carry, expected_ys = jax.lax.scan(_scan_step, jnp.array(0.0), XS, reverse=True)

    assert_allclose(carry, expected_carry)
    assert_allclose(ys, expected_ys)


def test_scan_matches_jax_lax_scan_when_driven_by_length_alone():
    """With no `xs`, `length` fixes the trip count and the scanned element is
    `None`. This is the idiom used to run a fixed number of inference steps."""

    def body(x, carry):
        assert x is None
        return carry + 1.0, carry

    (carry,), ys = pxf.scan(body, None, length=4)(jnp.array(0.0))
    expected_carry, expected_ys = jax.lax.scan(lambda c, _: (c + 1.0, c), jnp.array(0.0), None, length=4)

    assert_allclose(carry, expected_carry)
    assert_allclose(ys, expected_ys)


def test_scan_writes_back_a_param_mutated_in_the_body():
    """The caller must see the parameter as of the *last* iteration.

    `jax.lax.scan` threads state through the carry; pcx hides that by putting the
    tracked kwargs into the carry for you. If the write-back took the initial
    value, or a single step's, an inference loop would appear to run and change
    nothing.
    """
    param = pcx.Param(jnp.array(0.0))

    def body(x, carry, *, p):
        p.set(p.get() + x)
        return carry * 0.5 + x, p.get()

    (carry,), ys = pxf.scan(body, XS)(jnp.array(0.0), p=param)

    def reference(state, x):
        carry, acc = state
        acc = acc + x
        return (carry * 0.5 + x, acc), acc

    (expected_carry, expected_acc), expected_ys = jax.lax.scan(reference, (jnp.array(0.0), jnp.array(0.0)), XS)

    assert_allclose(carry, expected_carry)
    assert_allclose(ys, expected_ys)
    assert_allclose(param.get(), expected_acc)


def test_scan_does_not_write_back_a_param_passed_positionally():
    """Positional arguments are pure jax values: the loop still computes with the
    mutated value, but the caller's object is left alone."""
    start = jnp.array(0.0)
    param = pcx.Param(start)

    def body(x, carry, p):
        p.set(p.get() + x)
        return (carry * 0.5 + x, p), p.get()

    (carry, _), ys = pxf.scan(body, XS)(jnp.array(0.0), param)

    def reference(state, x):
        carry, acc = state
        acc = acc + x
        return (carry * 0.5 + x, acc), acc

    (expected_carry, _), expected_ys = jax.lax.scan(reference, (jnp.array(0.0), start), XS)

    assert_allclose(carry, expected_carry)
    assert_allclose(ys, expected_ys)
    assert_allclose(param.get(), start)


def test_scan_updates_a_shared_param_exactly_once():
    """A `Param` reachable under two kwargs is one parameter. Applying the loop's
    accumulated update once per reference would double it — here `+10.0` would
    become `+20.0` — with nothing raising."""
    param = pcx.Param(jnp.array(0.0))
    seen = {}

    def body(x, carry, *, a, b):
        seen["same_object"] = a is b
        a.set(a.get() + x)
        return carry, b.get()

    _, ys = pxf.scan(body, XS)(jnp.array(0.0), a=param, b=param)

    assert seen["same_object"] is True, "a shared Param must arrive as one object, not two copies"
    assert_allclose(param.get(), XS.sum())
    assert_allclose(ys, jnp.cumsum(XS))


def test_jit_of_scan_matches_the_unnested_form():
    """Composing transforms may not change any number: jitting a scan is an
    optimisation, not a different computation."""

    def body(x, carry, *, p):
        p.set(p.get() + x)
        return carry * 0.5 + x, p.get()

    plain_param = pcx.Param(jnp.array(0.0))
    (plain_carry,), plain_ys = pxf.scan(body, XS)(jnp.array(0.0), p=plain_param)

    jitted_param = pcx.Param(jnp.array(0.0))
    (jitted_carry,), jitted_ys = pxf.jit()(pxf.scan(body, XS))(jnp.array(0.0), p=jitted_param)

    expected_carry, _ = jax.lax.scan(_scan_step, jnp.array(0.0), XS)

    assert_allclose(plain_carry, expected_carry)
    assert_allclose(jitted_carry, expected_carry)
    assert_allclose(jitted_ys, plain_ys)
    assert_allclose(jitted_param.get(), plain_param.get())
    assert_allclose(jitted_param.get(), XS.sum())


def test_value_and_grad_through_scan_matches_jax_grad_of_lax_scan():
    """Gradients must flow through the loop, not just values.

    `jax.lax.scan` has its own transpose rule; a wrapper that broke the link
    between the tracked kwargs and the carry would still produce the right forward
    number while returning a zero or truncated gradient — the failure mode that
    makes a model silently stop learning.
    """
    w = jnp.array(0.5)
    init = jnp.array(1.0)
    param = pcx.Param(w)

    def body(x, carry, *, p):
        return carry * p.get() + x, None

    @pxf.value_and_grad(pxu.M(pcx.Param).to((False, True)))
    def loss(init, *, p):
        (carry,), _ = pxf.scan(body, XS)(init, p=p)
        return carry.sum()

    value, grads = loss(init, p=param)

    def reference(w, init):
        carry, _ = jax.lax.scan(lambda c, x: (c * w + x, None), init, XS)
        return carry.sum()

    expected_value, expected_grad = jax.value_and_grad(reference)(w, init)

    assert_allclose(value, expected_value)
    assert_allclose(grads["p"].get(), expected_grad)


@pytest.mark.bug(
    "BUGS.md#20: the Scan docstring example returns the final carry (20), not the '[0, 1, 3, 6, 10], None' it documents"
)
def test_scan_docstring_example_produces_its_documented_output():
    """The `Scan` class docstring carries the library's only executable example of
    the transform, and it is what a user will copy:

        def f(x, count):
            count = count + x
            return (count + x,), None

        Scan(f, xs=jax.numpy.arange(5))(0)  # [0, 1, 3, 6, 10], None

    Two things are wrong with the annotation. The body adds `x` twice per step, so
    the carry runs 0, 2, 6, 12, 20 rather than the cumulative sum shown; and the
    first element of the result is the final carry, not a per-step sequence — the
    per-step slot is the `None` the body returns. The documented output is
    therefore unreachable by any scan, and a reader calibrates their mental model
    of the argument convention on it.
    """
    from pcx.functional import Scan

    def f(x, count):
        count = count + x
        return (count + x,), None

    result, ys = Scan(f, xs=jnp.arange(5))(0)

    assert ys is None
    assert_allclose(jnp.asarray(result), jnp.array([0, 1, 3, 6, 10]))


# while_loop ###########################################################################################################


def test_while_loop_matches_jax_lax_while_loop():
    """A dynamic trip count is the whole point of `while_loop`; the wrapper must
    stop on exactly the same iteration as raw jax and return the same state."""

    def body(x, count):
        return x * 2.0 + 1.0, count + 1

    def cond_fun(x, count):
        return count < 4

    x, count = pxf.while_loop(body, cond_fun)(jnp.array(0.0), 0)
    expected = jax.lax.while_loop(
        lambda v: v[1] < 4,
        lambda v: (v[0] * 2.0 + 1.0, v[1] + 1),
        (jnp.array(0.0), 0),
    )

    assert_allclose(x, expected[0])
    assert_allclose(count, expected[1])


def test_while_loop_writes_back_a_param_mutated_in_the_body():
    """The caller sees the parameter as of the final iteration, exactly as raw jax
    would report the corresponding carry element."""
    param = pcx.Param(jnp.array(1.0))

    def body(count, *, p):
        p.set(p.get() * 2.0)
        return count + 1

    def cond_fun(count, *, p):
        return count < 3

    (count,) = pxf.while_loop(body, cond_fun)(0, p=param)
    expected = jax.lax.while_loop(
        lambda v: v[0] < 3,
        lambda v: (v[0] + 1, v[1] * 2.0),
        (0, jnp.array(1.0)),
    )

    assert_allclose(count, expected[0])
    assert_allclose(param.get(), expected[1])


def test_while_loop_condition_can_read_tracked_keyword_params():
    """`_cond_fn` is documented to "look at both args and kwargs", so a loop may
    terminate on a model's own state rather than a counter. If the condition only
    ever saw the positional arguments, this loop would never stop."""
    param = pcx.Param(jnp.array(1.0))

    def body(count, *, p):
        p.set(p.get() * 2.0)
        return count + 1

    def cond_fun(count, *, p):
        return jnp.all(p.get() < 8.0)

    (count,) = pxf.while_loop(body, cond_fun)(0, p=param)
    expected = jax.lax.while_loop(
        lambda v: jnp.all(v[1] < 8.0),
        lambda v: (v[0] + 1, v[1] * 2.0),
        (0, jnp.array(1.0)),
    )

    assert_allclose(count, expected[0])
    assert_allclose(param.get(), expected[1])


def test_while_loop_does_not_write_back_a_param_passed_positionally():
    """Positional arguments stay pure."""
    start = jnp.array(1.0)
    param = pcx.Param(start)

    def body(count, p):
        p.set(p.get() * 2.0)
        return count + 1, p

    def cond_fun(count, p):
        return count < 3

    count, _ = pxf.while_loop(body, cond_fun)(0, param)

    assert_allclose(count, 3)
    assert_allclose(param.get(), start)


def test_while_loop_updates_a_shared_param_exactly_once():
    """One object, one write-back, however many kwargs point at it."""
    param = pcx.Param(jnp.array(0.0))
    seen = {}

    def body(count, *, a, b):
        seen["same_object"] = a is b
        a.set(a.get() + 1.0)
        return count + 1

    def cond_fun(count, *, a, b):
        return count < 3

    pxf.while_loop(body, cond_fun)(0, a=param, b=param)

    assert seen["same_object"] is True, "a shared Param must arrive as one object, not two copies"
    assert_allclose(param.get(), jnp.array(3.0))


def test_jit_of_while_loop_matches_the_unnested_form():
    """Jitting a while loop is an optimisation, not a different computation."""

    def body(count, *, p):
        p.set(p.get() * 2.0)
        return count + 1

    def cond_fun(count, *, p):
        return count < 3

    plain_param = pcx.Param(jnp.array(1.0))
    (plain_count,) = pxf.while_loop(body, cond_fun)(0, p=plain_param)

    jitted_param = pcx.Param(jnp.array(1.0))
    (jitted_count,) = pxf.jit()(pxf.while_loop(body, cond_fun))(0, p=jitted_param)

    assert_allclose(plain_count, 3)
    assert_allclose(jitted_count, 3)
    assert_allclose(jitted_param.get(), plain_param.get())
    assert_allclose(jitted_param.get(), jnp.array(8.0))


# cond #################################################################################################################


def _true_branch(x):
    return jnp.tanh(x) + 1.0


def _false_branch(x):
    return jnp.tanh(x) - 1.0


@pytest.mark.parametrize("predicate", [True, False])
def test_cond_matches_jax_lax_cond_on_both_branches(predicate):
    """`Cond` passes its branches to `jax.lax.cond` in constructor order, so the
    *first* one must be the one taken when the predicate is true. Swapping them is
    a silent, total inversion of the program's logic."""
    x = jnp.array([0.5, -1.5])

    result = pxf.cond(_true_branch, _false_branch)(predicate, x)
    expected = jax.lax.cond(predicate, _true_branch, _false_branch, x)

    assert_allclose(result, expected)


@pytest.mark.parametrize("predicate", [True, False])
def test_cond_writes_back_only_the_mutation_of_the_taken_branch(predicate):
    """Both branches are traced but only one runs. If the write-back picked up the
    untaken branch's value — or merged the two — the parameter would end up in a
    state no branch ever produced."""
    start = jnp.array(1.0)
    param = pcx.Param(start)

    def add(x, *, p):
        p.set(p.get() + x)
        return p.get() * 10.0

    def multiply(x, *, p):
        p.set(p.get() * x)
        return p.get() * 10.0

    x = jnp.array(2.0)
    result = pxf.cond(add, multiply)(predicate, x, p=param)

    expected_value = jax.lax.cond(predicate, lambda v: v + x, lambda v: v * x, start)

    assert_allclose(param.get(), expected_value)
    assert_allclose(result, expected_value * 10.0)


def test_cond_does_not_write_back_a_param_passed_positionally():
    """Positional arguments stay pure."""
    start = jnp.array(1.0)
    param = pcx.Param(start)

    def add(x, p):
        p.set(p.get() + x)
        return p.get()

    def multiply(x, p):
        p.set(p.get() * x)
        return p.get()

    result = pxf.cond(add, multiply)(True, jnp.array(2.0), param)

    assert_allclose(result, jnp.array(3.0))
    assert_allclose(param.get(), start)


def test_cond_updates_a_shared_param_exactly_once():
    """One object, one write-back."""
    param = pcx.Param(jnp.array(1.0))
    seen = {}

    def add(*, a, b):
        seen["same_object"] = a is b
        a.set(a.get() + 1.0)
        return b.get()

    def noop(*, a, b):
        return b.get()

    result = pxf.cond(add, noop)(True, a=param, b=param)

    assert seen["same_object"] is True, "a shared Param must arrive as one object, not two copies"
    assert_allclose(param.get(), jnp.array(2.0))
    assert_allclose(result, jnp.array(2.0))


def test_jit_of_cond_matches_the_unnested_form():
    """Jitting a branch is an optimisation, not a different computation."""

    def add(x, *, p):
        p.set(p.get() + x)
        return p.get()

    def multiply(x, *, p):
        p.set(p.get() * x)
        return p.get()

    x = jnp.array(2.0)

    plain_param = pcx.Param(jnp.array(1.0))
    plain = pxf.cond(add, multiply)(True, x, p=plain_param)

    jitted_param = pcx.Param(jnp.array(1.0))
    jitted = pxf.jit()(pxf.cond(add, multiply))(True, x, p=jitted_param)

    assert_allclose(plain, jnp.array(3.0))
    assert_allclose(jitted, plain)
    assert_allclose(jitted_param.get(), plain_param.get())


def test_value_and_grad_through_cond_matches_jax_grad_of_the_taken_branch():
    """Differentiating a branch must give the taken branch's gradient and nothing
    from the other one. `jax.lax.cond` traces both, so a wrapper that summed or
    swapped their cotangents would still return the right forward value."""
    w = jnp.array([1.0, 2.0])
    x = jnp.array([3.0, 4.0])
    param = pcx.Param(w)

    def multiply(x, *, p):
        return (p.get() * x).sum()

    def divide(x, *, p):
        return (p.get() / x).sum()

    @pxf.value_and_grad(pxu.M(pcx.Param).to((False, True)))
    def loss(x, *, p):
        return pxf.cond(multiply, divide)(True, x, p=p)

    value, grads = loss(x, p=param)
    expected_value, expected_grad = jax.value_and_grad(lambda v, x: (v * x).sum())(w, x)

    assert_allclose(value, expected_value)
    assert_allclose(grads["p"].get(), expected_grad)


# switch ###############################################################################################################


def _branch_0(x):
    return x + 10.0


def _branch_1(x):
    return x * 10.0


def _branch_2(x):
    return x - 10.0


SWITCH_BRANCHES = (_branch_0, _branch_1, _branch_2)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_switch_matches_jax_lax_switch_for_every_index(index):
    """`Switch` must dispatch on the index in the order the branches were given —
    an off-by-one here silently runs the wrong branch for every input."""
    x = jnp.array([2.0, -3.0])

    result = pxf.switch(SWITCH_BRANCHES)(index, x)
    expected = jax.lax.switch(index, SWITCH_BRANCHES, x)

    assert_allclose(result, expected)


@pytest.mark.bug(
    "#74: pxf.switch([...]) raises TypeError: _make_tuple leaves a list alone, collapsing all branches into one"
)
def test_switch_accepts_a_list_of_branches():
    """`pcx.functional.switch` is typed `branches: Sequence[...]` and mirrors
    `jax.lax.switch`, whose branches are conventionally written as a list — that
    is also how every `jax.lax.switch` example in the wild spells it.

    `_BaseTransform.__init__` normalises `fn` with `_make_tuple`, which is
    `x if isinstance(x, tuple) else (x,)`: a list is not a tuple, so it is wrapped
    rather than expanded and the whole list becomes a single "function".
    `Switch._t` then calls `len(self.fn)` on that one wrapped callable and raises
    `TypeError: object of type 'function' has no len()`. Only the undocumented
    tuple spelling works.
    """
    x = jnp.array([2.0, -3.0])

    result = pxf.switch([_branch_0, _branch_1, _branch_2])(1, x)

    assert_allclose(result, jax.lax.switch(1, SWITCH_BRANCHES, x))


@pytest.mark.parametrize("index", [0, 1, 2])
def test_switch_writes_back_the_mutation_of_the_selected_branch(index):
    """Every branch is traced; only the selected one may reach the caller's
    parameter."""
    start = jnp.array(3.0)
    param = pcx.Param(start)

    def add(x, *, p):
        p.set(p.get() + x)
        return 0

    def multiply(x, *, p):
        p.set(p.get() * x)
        return 1

    def subtract(x, *, p):
        p.set(p.get() - x)
        return 2

    x = jnp.array(2.0)
    tag = pxf.switch((add, multiply, subtract))(index, x, p=param)

    expected = jax.lax.switch(index, (lambda v: v + x, lambda v: v * x, lambda v: v - x), start)

    assert tag == index, "switch must run the branch at the requested index"
    assert_allclose(param.get(), expected)


def test_switch_does_not_write_back_a_param_passed_positionally():
    """Positional arguments stay pure."""
    start = jnp.array(3.0)
    param = pcx.Param(start)

    def add(x, p):
        p.set(p.get() + x)
        return p.get()

    def subtract(x, p):
        p.set(p.get() - x)
        return p.get()

    result = pxf.switch((add, subtract))(0, jnp.array(2.0), param)

    assert_allclose(result, jnp.array(5.0))
    assert_allclose(param.get(), start)


def test_switch_updates_a_shared_param_exactly_once():
    """One object, one write-back."""
    param = pcx.Param(jnp.array(1.0))
    seen = {}

    def bump(*, a, b):
        seen["same_object"] = a is b
        a.set(a.get() + 1.0)
        return b.get()

    def noop(*, a, b):
        return b.get()

    result = pxf.switch((bump, noop))(0, a=param, b=param)

    assert seen["same_object"] is True, "a shared Param must arrive as one object, not two copies"
    assert_allclose(param.get(), jnp.array(2.0))
    assert_allclose(result, jnp.array(2.0))


def test_jit_of_switch_matches_the_unnested_form():
    """Jitting a switch is an optimisation, not a different computation."""

    def add(x, *, p):
        p.set(p.get() + x)
        return p.get()

    def subtract(x, *, p):
        p.set(p.get() - x)
        return p.get()

    x = jnp.array(2.0)

    plain_param = pcx.Param(jnp.array(3.0))
    plain = pxf.switch((add, subtract))(1, x, p=plain_param)

    jitted_param = pcx.Param(jnp.array(3.0))
    jitted = pxf.jit()(pxf.switch((add, subtract)))(1, x, p=jitted_param)

    assert_allclose(plain, jnp.array(1.0))
    assert_allclose(jitted, plain)
    assert_allclose(jitted_param.get(), plain_param.get())


# Randomness ###########################################################################################################
#
# These come last: the exception test can leave a tracer installed in the global
# RKG, and although conftest's autouse fixture reseeds it, ordering keeps any
# fallout away from the tests above.


def test_scan_advances_the_global_rkg_exactly_like_a_raw_jax_key_carry():
    """`RKG` rides through `jax.lax.scan` as part of the carry, so a draw per
    iteration must consume one split per iteration and leave the caller's global
    key advanced by exactly that many splits.

    If the key were not carried, every iteration would draw the *same* sample —
    the classic jax RNG mistake, and one that produces perfectly plausible-looking
    output.
    """
    pcx.RKG.seed(0)

    def body(_, total):
        return total + jax.random.normal(pcx.RKG(), ()), None

    (total,), _ = pxf.scan(body, None, length=3)(jnp.array(0.0))

    def reference(state, _):
        acc, key = state
        fresh = jax.random.split(key, 2)
        return (acc + jax.random.normal(fresh[1], ()), fresh[0]), None

    (expected_total, expected_key), _ = jax.lax.scan(reference, (jnp.array(0.0), jax.random.PRNGKey(0)), None, length=3)

    assert_allclose(total, expected_total)
    assert jnp.array_equal(pcx.RKG.key.get(), expected_key), "the global key must be advanced once per iteration"


def test_cond_advances_the_global_rkg_exactly_like_a_raw_jax_split():
    """The taken branch's key consumption must escape the `jax.lax.cond` and reach
    the caller's global generator."""
    pcx.RKG.seed(0)

    def draws(x):
        return x + jax.random.normal(pcx.RKG(), (2,))

    def also_draws(x):
        return x - jax.random.normal(pcx.RKG(), (2,))

    sample = pxf.cond(draws, also_draws)(True, jnp.zeros(2))

    fresh = jax.random.split(jax.random.PRNGKey(0), 2)
    assert_allclose(sample, jax.random.normal(fresh[1], (2,)))
    assert jnp.array_equal(pcx.RKG.key.get(), fresh[0]), "the global key must be advanced past the consumed one"


def test_global_rkg_holds_a_concrete_array_after_a_flow_function_raises():
    """`_BaseTransform` swaps `RKG.key` for the traced key on the way in and
    restores it on the way out, but the restore is a plain statement rather than a
    `finally`. A body that raises during tracing therefore leaves the tracer
    installed in module-level state.

    `pcx.RKG` is the default argument of every layer and Vode constructor, so from
    that moment the whole process builds models from a dead tracer and fails with
    an unrelated `UnexpectedTracerError` far from the real cause. A shape error in
    an inference loop must not poison the interpreter.

    Kept last in the file, since a failure here leaks global state.
    """
    pcx.RKG.seed(0)

    def body(_, total):
        raise ValueError("deliberate failure inside the loop body")

    with pytest.raises(ValueError, match="deliberate failure"):
        pxf.scan(body, None, length=2)(jnp.array(0.0))

    key_type = type(pcx.RKG.key.get()).__name__
    assert "Tracer" not in key_type, f"pcx.RKG.key holds a {key_type} after the failed call, not a concrete array"
