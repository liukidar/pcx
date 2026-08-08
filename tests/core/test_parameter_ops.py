"""Operator forwarding on `pcx.Param`.

`Param` wraps a `jax.Array` and explicitly forwards ~60 dunder methods to it
(`pcx/core/_parameter.py`, lines 128-289). The comment above that block states the
intent outright: Python looks up special methods on the class, not the instance, so
the transparent forwarding that `__getattr__` gives ordinary attributes has to be
written out by hand for operators.

That intent is a complete and independent oracle, and it is the only one used in
this file: **an operation on `Param(x)` must produce exactly what the same operation
on the bare array `x` produces** — same values, same dtype, same shape. Nothing here
is derived from what pcx happens to return.

Why it matters: a `Param` is what a user's model holds, so `w * x`, `w[i]`, `-w` and
`w.sum()` are written against parameters all over user code and throughout the
tutorials. A forwarding slip does not raise; it produces a plausible number of the
wrong dtype, or silently drops the wrapper, and the result flows straight into an
energy or a gradient.
"""

import itertools
import operator

import jax.numpy as jnp
import numpy as np
import pytest

import pcx

# Operands are chosen so that every operation below is well defined: strictly
# positive floats (so `//`, `%` and `**` have no sign or domain corner), and
# small non-negative integers for the bitwise and shift operators.
A = jnp.array([[3.0, 1.5], [2.0, 4.0]])
B = jnp.array([[1.0, 2.0], [0.5, 0.25]])

IA = jnp.array([[6, 3], [12, 5]], dtype=jnp.int32)
IB = jnp.array([[3, 1], [5, 2]], dtype=jnp.int32)

#: (operator, left, right) triples. The expected result of each is `op(left, right)`
#: evaluated on the bare arrays — never on a Param.
BINARY_OPS = [
    (operator.add, A, B),
    (operator.sub, A, B),
    (operator.mul, A, B),
    (operator.truediv, A, B),
    (operator.floordiv, A, B),
    (operator.mod, A, B),
    (operator.pow, A, B),
    (operator.matmul, A, B),
    (operator.lt, A, B),
    (operator.le, A, B),
    (operator.gt, A, B),
    (operator.ge, A, B),
    (operator.eq, A, B),
    (operator.ne, A, B),
    (operator.and_, IA, IB),
    (operator.or_, IA, IB),
    (operator.xor, IA, IB),
    (operator.lshift, IA, IB),
    (operator.rshift, IA, IB),
]

BINARY_IDS = [
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "pow",
    "matmul",
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "ne",
    "and",
    "or",
    "xor",
    "lshift",
    "rshift",
]


def assert_same_array(actual, expected, *, context: str) -> None:
    """The result must be indistinguishable from the bare-array result.

    dtype and shape are checked as well as the values: a wrapper that promoted
    `float32` to `float64`, or that broadcast differently, would still produce
    numbers that look right in a print.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)

    assert actual.shape == expected.shape, f"{context}: shape {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"{context}: dtype {actual.dtype} != {expected.dtype}"
    np.testing.assert_array_equal(actual, expected, err_msg=context)


# Binary operators #####################################################################


@pytest.mark.parametrize(("op", "left", "right"), BINARY_OPS, ids=BINARY_IDS)
def test_binary_operator_on_two_params_matches_the_bare_arrays(op, left, right):
    """`op(Param(a), Param(b))` must equal `op(a, b)`.

    This is the common case in a pcx model: both operands are parameters, so the
    forwarded method has to unwrap the right-hand side (via `pcx.get`) as well as
    the left. If it did not, jax would receive a `Param` as an operand and either
    raise or, worse, treat it as an opaque object.
    """
    assert_same_array(op(pcx.Param(left), pcx.Param(right)), op(left, right), context=f"{op.__name__}(Param, Param)")


@pytest.mark.parametrize(("op", "left", "right"), BINARY_OPS, ids=BINARY_IDS)
def test_binary_operator_with_a_bare_right_operand_matches_the_bare_arrays(op, left, right):
    """`op(Param(a), b)` must equal `op(a, b)`: a parameter has to interoperate with
    a plain array without the caller unwrapping it first."""
    assert_same_array(op(pcx.Param(left), right), op(left, right), context=f"{op.__name__}(Param, array)")


@pytest.mark.parametrize(("op", "left", "right"), BINARY_OPS, ids=BINARY_IDS)
def test_reflected_binary_operator_with_a_bare_left_operand_matches_the_bare_arrays(op, left, right):
    """`op(a, Param(b))` must equal `op(a, b)`.

    This is the reflected half, and it is the fragile one: jax's array operators
    return `NotImplemented` for an unrecognised right operand, so Python falls back
    to `Param.__r*__`. If a reflected method were missing or forwarded to the
    non-reflected one, `x - w` would silently compute `w - x` — a sign error with no
    error message anywhere.
    """
    assert_same_array(op(left, pcx.Param(right)), op(left, right), context=f"{op.__name__}(array, Param)")


@pytest.mark.parametrize(
    ("op", "scalar"),
    [
        (operator.add, 2.0),
        (operator.sub, 2.0),
        (operator.mul, 3.0),
        (operator.truediv, 4.0),
        (operator.floordiv, 2.0),
        (operator.mod, 2.0),
        (operator.pow, 2.0),
        (operator.lt, 2.0),
        (operator.ge, 2.0),
    ],
    ids=["add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "lt", "ge"],
)
def test_operators_against_a_python_scalar_match_the_bare_array(op, scalar):
    """Mixing a parameter with a Python number — `lr * grad`, `w ** 2` — must behave
    exactly as it does for the bare array, in both operand orders.

    Weak-typed Python scalars are the case where jax's promotion rules are most
    likely to be disturbed by a wrapper: a spurious cast to float64 (or to a weak
    type) here would change the dtype of a whole model.
    """
    assert_same_array(op(pcx.Param(A), scalar), op(A, scalar), context=f"{op.__name__}(Param, scalar)")
    assert_same_array(op(scalar, pcx.Param(A)), op(scalar, A), context=f"{op.__name__}(scalar, Param)")


def test_divmod_matches_the_bare_arrays_in_both_operand_orders():
    """`divmod` returns a *pair*, so it is the one forwarded operator whose result is
    not a single array; both halves have to survive the wrapper."""
    for actual, expected, context in (
        (divmod(pcx.Param(A), pcx.Param(B)), divmod(A, B), "divmod(Param, Param)"),
        (divmod(pcx.Param(A), B), divmod(A, B), "divmod(Param, array)"),
        (divmod(A, pcx.Param(B)), divmod(A, B), "divmod(array, Param)"),
    ):
        assert len(actual) == 2, f"{context}: expected a (quotient, remainder) pair"
        assert_same_array(actual[0], expected[0], context=f"{context} quotient")
        assert_same_array(actual[1], expected[1], context=f"{context} remainder")


# Unary operators ######################################################################


@pytest.mark.parametrize(
    ("op", "value"),
    [
        (operator.neg, jnp.array([1.0, -2.5])),
        (operator.pos, jnp.array([1.0, -2.5])),
        (operator.abs, jnp.array([1.0, -2.5])),
        (operator.invert, jnp.array([6, -3], dtype=jnp.int32)),
        (operator.invert, jnp.array([True, False])),
    ],
    ids=["neg", "pos", "abs", "invert_int", "invert_bool"],
)
def test_unary_operator_matches_the_bare_array(op, value):
    """`-w`, `abs(w)` and `~mask` are written directly on parameters throughout the
    library (gradient descent is literally `w - lr * g`), so the unary forwards have
    to be exact."""
    assert_same_array(op(pcx.Param(value)), op(value), context=f"{op.__name__}(Param)")


def test_round_matches_the_bare_array():
    """`round(w)` and `round(w, ndigits)` both forward to the array's `__round__`."""
    value = jnp.array([1.234, 2.5, -0.75])

    assert_same_array(round(pcx.Param(value)), round(value), context="round(Param)")
    assert_same_array(round(pcx.Param(value), 1), round(value, 1), context="round(Param, 1)")


# Array protocol and container behaviour ###############################################


@pytest.mark.parametrize("attribute", ["shape", "dtype", "ndim"], ids=["shape", "dtype", "ndim"])
def test_array_metadata_properties_match_the_bare_array(attribute):
    """`shape`, `dtype` and `ndim` are declared as explicit properties rather than
    left to `__getattr__`. They are the metadata every reshape, mask and batching
    decision in the library reads, so they must describe the wrapped array."""
    assert getattr(pcx.Param(A), attribute) == getattr(A, attribute)


@pytest.mark.parametrize(
    "index",
    [0, -1, slice(1, None), (0, 1), (Ellipsis, 0)],
    ids=["int", "negative", "slice", "tuple", "ellipsis"],
)
def test_indexing_matches_the_bare_array(index):
    """`__getitem__` is forwarded, so every jax indexing form — integer, negative,
    slice, multi-axis tuple, ellipsis — must give the array's own answer. Slicing a
    parameter is how a batch is taken apart, so a silent off-by-one here would
    misalign inputs and targets."""
    assert_same_array(pcx.Param(A)[index], A[index], context=f"Param[{index!r}]")


@pytest.mark.bug(
    "iterating a Param never terminates: no __iter__, and the __getitem__ fallback never stops because "
    "jax clamps an out-of-bounds index instead of raising IndexError"
)
def test_iterating_a_param_yields_exactly_the_elements_of_the_array():
    """`for row in w` must walk the leading axis exactly as `for row in w.get()` does,
    and must stop at the end of it.

    `Param` defines no `__iter__`, so Python falls back to the legacy sequence
    protocol: call `__getitem__` with 0, 1, 2, ... until it raises `IndexError`. jax
    never raises — an out-of-bounds index is *clamped* to the last element — so the
    fallback yields the final row forever. Iterating a parameter hangs the process
    and grows without bound, and it is a completely ordinary thing to write, since
    iterating the wrapped array works fine.

    The iterator is therefore truncated at `len(A) + 1` here rather than drained:
    a correct iterator stops after `len(A)` elements, and taking one more from it
    yields nothing.
    """
    rows = list(itertools.islice(iter(pcx.Param(A)), len(A) + 1))

    assert len(rows) == len(A), f"iteration yielded {len(rows)} elements for an array of length {len(A)}"
    for i, (actual, expected) in enumerate(zip(rows, list(A), strict=True)):
        assert_same_array(actual, expected, context=f"row {i}")


@pytest.mark.bug("Param forwards __getitem__ and .shape but defines no __len__, so len(param) raises TypeError")
def test_len_matches_the_bare_array():
    """`len(w)` must be the size of the leading axis, as it is for the array.

    `Param` deliberately enumerates the dunders it forwards — including
    `__getitem__`, which makes a parameter iterable — but `__len__` is not among
    them, and `__getattr__` is never consulted for special methods (the comment in
    the source says exactly this). So a parameter is a sequence you can index and
    iterate but not measure: `len(w)` raises `TypeError: object of type 'Param' has
    no len()`, and so does anything built on it (`np.array(list)`-style code,
    `zip(..., strict=True)` over a parameter, a `batch_size = len(x)` line).
    """
    assert len(pcx.Param(A)) == len(A)


@pytest.mark.bug("Param forwards no __float__/__int__, so float(param) on a scalar parameter raises TypeError")
def test_scalar_conversions_match_the_bare_array():
    """`float(loss)` and `int(count)` on a zero-dimensional parameter must give the
    same Python number the wrapped array gives.

    This is the shape of every logging line ever written — `print(f"{float(loss):.3f}")`,
    `history.append(float(energy))`. `Param` forwards `__array__` but neither
    `__float__` nor `__int__`, and special methods are never routed through
    `__getattr__`, so the conversion fails with `TypeError: float() argument must be
    a string or a real number, not 'Param'` and the user has to remember `.get()`.
    """
    scalar = jnp.array(2.5)
    count = jnp.array(7, dtype=jnp.int32)

    assert float(pcx.Param(scalar)) == float(scalar)
    assert int(pcx.Param(count)) == int(count)


def test_numpy_conversion_matches_the_bare_array():
    """`__array__` is forwarded so a parameter can be handed to numpy, matplotlib or
    any library that speaks the buffer protocol. Evaluation and plotting code does
    `np.asarray(param)` constantly."""
    assert_same_array(np.asarray(pcx.Param(A)), np.asarray(A), context="np.asarray(Param)")


def test_numpy_conversion_honours_an_explicit_dtype():
    """`np.asarray(param, dtype=...)` must cast exactly as it does for the array;
    `__array__` takes the dtype argument precisely so numpy can request one."""
    assert_same_array(np.asarray(pcx.Param(A), dtype=np.float64), np.asarray(A, dtype=np.float64), context="dtype")


# Attribute forwarding #################################################################


@pytest.mark.parametrize(
    ("call", "name"),
    [
        (lambda x: x.sum(), "sum"),
        (lambda x: x.mean(), "mean"),
        (lambda x: x.min(), "min"),
        (lambda x: x.max(), "max"),
        (lambda x: x.reshape(4), "reshape"),
        (lambda x: x.ravel(), "ravel"),
        (lambda x: x.transpose(), "transpose"),
        (lambda x: x.astype(jnp.int32), "astype"),
        (lambda x: x.sum(axis=0), "sum_axis"),
        (lambda x: x.T, "T"),
        (lambda x: x.at[0, 0].set(9.0), "at_set"),
    ],
)
def test_array_methods_reached_through_getattr_match_the_bare_array(call, name):
    """`__getattr__` forwards every non-special attribute to the wrapped array, so
    `param.sum()` is `param.get().sum()`.

    This is what lets a `Param` be dropped into code written for arrays. If the
    forward returned a bound method of the wrong object — or shadowed a name that
    `Param` itself happens to define — the call would still return an array and the
    substitution would be silently wrong.
    """
    assert_same_array(call(pcx.Param(A)), call(A), context=f"Param.{name}")


def test_getattr_raises_attribute_error_for_a_name_the_array_does_not_have():
    """Forwarding must not invent attributes: an unknown name has to fail as an
    `AttributeError`, which is what `hasattr` and every duck-typing check in the
    library (`hasattr(x, "inference")` in `Module.train`, for one) rely on."""
    with pytest.raises(AttributeError):
        getattr(pcx.Param(A), "definitely_not_an_array_attribute")  # noqa: B009


# repr #################################################################################


def test_repr_of_an_array_param_reports_its_shape_and_dtype():
    """`repr` is what a user sees when they print a model — `BaseModule.__repr__`
    builds its whole listing out of parameter reprs. It has to identify the class and
    describe the array without dumping its contents, or printing a real model becomes
    unusable."""
    assert repr(pcx.Param(jnp.ones((2, 3), dtype=jnp.float32))) == "Param([2,3], float32)"


def test_repr_of_a_subclass_names_the_subclass():
    """Printed models are read to tell `VodeParam` from `LayerParam`, so the concrete
    class name — not `Param` — must appear."""

    class _WeightParam(pcx.Param):
        pass

    assert repr(_WeightParam(jnp.zeros((5,)))) == "_WeightParam([5], float32)"


def test_repr_of_an_empty_param_does_not_raise():
    """`Param(None)` is the normal state of a cleared parameter (`clear_params` sets
    every selected value to `None`), so printing a model between steps must fall back
    to the value's own repr instead of asking a `None` for its shape."""
    assert repr(pcx.Param(None)) == "Param(None)"


# In-place operators ###################################################################
#
# `__iadd__`, `__isub__` and `__imul__` are covered by
# tests/core/test_parameter.py::test_in_place_arithmetic_mutates_the_same_object,
# together with the known `/=` defect (BUGS.md#3). The remaining augmented
# assignments are below.


@pytest.mark.bug(
    "BUGS.md#3, generalised: Param defines only __iadd__/__isub__/__imul__, so //= %= **= @= &= |= ^= <<= >>= "
    "all rebind the name to a bare jax.Array and the parameter silently keeps its old value"
)
@pytest.mark.parametrize(
    ("op", "start", "other", "expected"),
    [
        (operator.ifloordiv, jnp.array(7.0), 2.0, jnp.array(3.0)),
        (operator.imod, jnp.array(7.0), 2.0, jnp.array(1.0)),
        (operator.ipow, jnp.array(3.0), 2.0, jnp.array(9.0)),
        (operator.imatmul, jnp.eye(2) * 2.0, jnp.eye(2) * 3.0, jnp.eye(2) * 6.0),
        (operator.iand, jnp.array(6, dtype=jnp.int32), 3, jnp.array(2, dtype=jnp.int32)),
        (operator.ior, jnp.array(6, dtype=jnp.int32), 3, jnp.array(7, dtype=jnp.int32)),
        (operator.ixor, jnp.array(6, dtype=jnp.int32), 3, jnp.array(5, dtype=jnp.int32)),
        (operator.ilshift, jnp.array(6, dtype=jnp.int32), 2, jnp.array(24, dtype=jnp.int32)),
        (operator.irshift, jnp.array(6, dtype=jnp.int32), 1, jnp.array(3, dtype=jnp.int32)),
    ],
    ids=["ifloordiv", "imod", "ipow", "imatmul", "iand", "ior", "ixor", "ilshift", "irshift"],
)
def test_every_augmented_assignment_mutates_the_same_object(op, start, other, expected):
    """`p @= m` must update `p` in place and leave the caller holding the *same*
    `Param` — the invariant the whole library rests on.

    pcx tracks state by object identity: `tree_ref` deduplicates on `id()`, and every
    transform writes its results back into the caller's parameter objects. When the
    augmented assignment falls through to the plain binary operator, Python rebinds
    the local name to a bare `jax.Array` while the module that owns the parameter
    still points at the untouched original. The update is discarded in silence, and
    the local variable is no longer a parameter at all.

    BUGS.md#3 records this for `/=` only; the same root cause covers the nine
    operators here, none of which has an in-place implementation.
    """
    param = pcx.Param(start)
    original = param

    result = op(param, other)

    assert result is original, f"{op.__name__} replaced the Param with a {type(result).__name__}"
    assert_same_array(original.get(), expected, context=op.__name__)
