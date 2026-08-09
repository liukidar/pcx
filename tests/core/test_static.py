"""Contracts of `pcx.static`.

A `StaticParam` is how a module carries things jax cannot trace — a callable, a
string, a shape, a mode flag — while remaining a pytree. Its defining property is
negative: it must contribute *nothing* to the dynamic side. Its payload lives in
the treedef instead, which is what makes it act as a compile-time constant: jax
compares treedefs to decide whether a cached executable may be reused, so a
change to a static payload has to force a retrace.

Expectations come from `pcx/core/_static.py`'s documented behaviour and from how
jax defines aux data, not from observed output.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from conftest import assert_allclose, count_leaves

import pcx


@pytest.mark.parametrize(
    "payload",
    [3, "text", (1, 2, 3), {"a": 1}, len, None],
    ids=["int", "str", "tuple", "dict", "callable", "none"],
)
def test_static_contributes_no_dynamic_leaves(payload):
    """If a static payload leaked into the leaves, jax would try to trace a
    string or a function and fail — and an optimiser would try to take a
    gradient step on it. Zero leaves is the entire point of the wrapper."""
    assert count_leaves(pcx.static(payload)) == 0


def test_static_payload_is_readable_through_get():
    payload = {"a": 1}

    assert pcx.static(payload).get() is payload


def test_static_is_idempotent():
    """`static` is applied defensively at many call sites; wrapping twice would
    produce a `StaticParam` whose payload is a `StaticParam`, and every
    `.get()` in the library would then return the wrapper instead of the value."""
    s = pcx.static("payload")

    assert pcx.static(s) is s


def test_static_payload_travels_in_the_treedef():
    """Two statics holding different payloads must be structurally different
    pytrees. This is what makes a static a compile-time constant."""
    assert jtu.tree_structure(pcx.static("a")) != jtu.tree_structure(pcx.static("b"))


def test_equal_static_payloads_give_equal_treedefs():
    """The converse: two modules configured identically must share a treedef, or
    jit would recompile for every freshly constructed instance."""
    assert jtu.tree_structure(pcx.static("a")) == jtu.tree_structure(pcx.static("a"))


def test_static_roundtrips_through_flatten_and_unflatten():
    """The payload has to survive a transform boundary, since it is carried
    purely by the treedef and never by a leaf."""
    s = pcx.static({"kind": "conv", "stride": 2})

    leaves, treedef = jtu.tree_flatten(s)
    rebuilt = jtu.tree_unflatten(treedef, leaves)

    assert leaves == []
    assert type(rebuilt) is type(s)
    assert rebuilt.get() == {"kind": "conv", "stride": 2}


def test_static_forwards_calls_to_its_payload():
    """Layers store their activation function as a static and call it directly;
    the wrapper is documented to be usable "as if it were the value itself"."""
    f = pcx.static(lambda x: x + 1)

    assert f(41) == 42


def test_a_module_holding_only_statics_has_no_dynamic_leaves():
    """A configuration-only module must be free to pass through a transform
    without contributing anything to trace."""

    class Config(pcx.Module):
        def __init__(self):
            super().__init__()
            self.name = pcx.static("cfg")
            self.stride = pcx.static(2)

    assert count_leaves(Config()) == 0


def test_mutating_a_static_inside_a_trace_does_not_escape():
    """Documented behaviour: "each change to a static parameter is temporary and
    does not affect the original value outside of a transformation".

    It has to be, and not merely by convention: inside a trace the module is
    rebuilt from the treedef, so a write there lands on a throwaway copy. If it
    escaped, the mutation would happen once at trace time and then silently never
    again on subsequent cached calls — the value would depend on whether the
    function was recompiled.
    """
    s = pcx.static("original")

    def f(x, param):
        param.set("mutated")
        return x + 1.0

    jax.jit(f)(jnp.array(1.0), s)

    assert s.get() == "original"


def test_mutating_a_static_outside_a_trace_changes_the_treedef():
    """Outside a transform a static is ordinary mutable configuration, and the
    change must be visible to jax as a structural change — otherwise the next
    call would reuse an executable compiled against the old payload."""
    s = pcx.static(2)
    before = jtu.tree_structure(s)

    s.set(3)

    assert jtu.tree_structure(s) != before


def test_changing_a_static_payload_retraces_the_jitted_function():
    """The observable consequence of the payload living in the treedef: a jitted
    function called with a different static must recompute, not return a stale
    result from the cache keyed on the old payload."""

    def scale(x, factor):
        return x * factor.get()

    jitted = jax.jit(scale)
    x = jnp.array(2.0)

    first = jitted(x, pcx.static(2.0))
    second = jitted(x, pcx.static(5.0))

    assert_allclose(first, 4.0)
    assert_allclose(second, 10.0)


def test_static_is_not_differentiated():
    """A static must be invisible to `jax.grad`: it has no leaves, so there is
    nothing to differentiate with respect to, and any attempt to treat it as a
    dynamic value would raise on a non-array payload."""
    mode = pcx.static("train")

    grad = jax.grad(lambda x, m: x * (2.0 if m.get() == "train" else 1.0))(jnp.array(3.0), mode)

    assert_allclose(grad, 2.0)
