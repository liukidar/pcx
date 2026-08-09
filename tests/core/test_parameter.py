"""Contracts of the parameter wrapper.

`Param` is the object every other part of pcx is built on: it is the only thing
jax is allowed to see as dynamic, and it is a *mutable reference* — the library
tracks state by mutating the same object rather than by threading a new one
through a functional pipeline. Two properties therefore carry the whole design:
a parameter must flatten to exactly one dynamic leaf, and every in-place update
must keep the same Python object alive.

Expectations below come from the docstrings in `pcx/core/_parameter.py`, from
Python's data model (what `+=`, `in`, `d[k]` mean), and from the semantics of a
jax pytree — never from what the implementation happens to return.
"""

import operator

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from conftest import assert_allclose, count_leaves

import pcx


class _CustomParam(pcx.Param):
    """A user-defined parameter type, declared exactly as a user would declare one."""


def test_param_flattens_to_exactly_one_dynamic_leaf():
    """The count of dynamic leaves is what every jax transform sizes its work by.
    A parameter that produced two leaves would double every gradient tree; one
    that produced none would silently drop the parameter from optimisation."""
    p = pcx.Param(jnp.ones(3))

    assert count_leaves(p) == 1


def test_param_leaf_is_the_wrapped_value_itself():
    """The wrapper must be transparent: what jax differentiates and what the user
    stored have to be the same array, not a copy or a transformed view."""
    value = jnp.arange(4.0)
    p = pcx.Param(value)

    (leaf,) = jtu.tree_leaves(p)

    assert leaf is value


@pytest.mark.parametrize("cls", [pcx.Param, _CustomParam])
def test_unflatten_rebuilds_the_same_concrete_class(cls):
    """jax rebuilds a pytree from its treedef after every transform. If the
    rebuilt object were a plain `Param` rather than the user's subclass, masks
    and filters written as `isinstance(x, MyParam)` would stop matching the
    moment a value came back out of `jit`."""
    p = cls(jnp.ones(2))

    rebuilt = jtu.tree_unflatten(*jtu.tree_flatten(p)[::-1])

    assert type(rebuilt) is cls


def test_unflatten_restores_the_wrapped_value():
    """A round-trip through the pytree machinery must be value-preserving,
    otherwise every transform boundary would corrupt state."""
    p = pcx.Param(jnp.array([1.0, 2.0, 3.0]))

    leaves, treedef = jtu.tree_flatten(p)
    rebuilt = jtu.tree_unflatten(treedef, leaves)

    assert_allclose(rebuilt.get(), jnp.array([1.0, 2.0, 3.0]))


def test_param_subclass_is_a_pytree_without_any_registration():
    """`_BaseParamMeta` registers every subclass on class creation, so a user
    defining a parameter type never calls `register_pytree_node`. If that broke,
    the subclass would be treated as an opaque leaf and its array would never
    reach the transform."""

    class LateDeclaredParam(pcx.Param):
        pass

    p = LateDeclaredParam(jnp.ones(5))

    assert count_leaves(p) == 1
    assert jtu.tree_leaves(p)[0].shape == (5,)


def test_tree_map_over_a_subclass_returns_that_subclass():
    """`tree_map` is how optimisers and masks rewrite parameter trees; the type
    of the parameter must survive that rewrite."""
    p = _CustomParam(jnp.ones(3))

    doubled = jtu.tree_map(lambda x: x * 2, p)

    assert type(doubled) is _CustomParam
    assert_allclose(doubled.get(), jnp.full((3,), 2.0))


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (operator.iadd, 6.0),
        (operator.isub, 2.0),
        (operator.imul, 8.0),
        (operator.itruediv, 2.0),
    ],
    ids=["iadd", "isub", "imul", "itruediv"],
)
def test_in_place_arithmetic_mutates_the_same_object(op, expected):
    """`p += 1` must update `p` in place and return the *same* object.

    pcx tracks state by identity: `tree_ref` deduplicates on `id()`, and a
    transform writes results back into the caller's parameter objects. An
    operator that returned a new object — or worse, a bare `jax.Array` — would
    silently detach the parameter from the module that holds it, so the update
    would be lost and the module would keep a raw array where a `Param` belongs.
    """
    p = pcx.Param(jnp.array(4.0))
    original = p

    result = op(p, 2.0)

    assert result is original, f"{op.__name__} replaced the Param instead of mutating it"
    assert_allclose(original.get(), expected)


def test_in_place_update_is_visible_through_every_alias():
    """Aliasing is the point of a mutable reference: a parameter shared between
    two modules must show one update, not two divergent copies."""
    p = pcx.Param(jnp.array(1.0))
    alias = p

    p += 1.0

    assert alias is p
    assert_allclose(alias.get(), 2.0)


@pytest.mark.parametrize(
    "make_param",
    [
        lambda: pcx.Param(jnp.ones(3)),
        lambda: pcx.Param(jnp.array(0.0)),
        lambda: pcx.ParamDict({"a": jnp.ones(1)}),
    ],
    ids=["array", "scalar", "dict"],
)
def test_truth_testing_a_parameter_raises(make_param):
    """`if param:` is a bug the author cannot see: for a scalar it would silently
    test the value, for an array it would raise deep inside jax. The guardrail
    turns it into an immediate, explicit TypeError at the point of the mistake."""
    with pytest.raises(TypeError, match="can not be used as Python bool"):
        bool(make_param())


def test_get_unwraps_a_parameter():
    """`get` is the coercion used everywhere a value may or may not be wrapped."""
    value = jnp.array([1.0, 2.0])

    assert pcx.get(pcx.Param(value)) is value


@pytest.mark.parametrize("value", [3, 3.0, "text", None, (1, 2)], ids=["int", "float", "str", "none", "tuple"])
def test_get_passes_plain_values_through_untouched(value):
    """The documented contract: a non-parameter is returned as is, so callers can
    apply `get` unconditionally."""
    assert pcx.get(value) is value


def test_set_writes_into_the_parameter_and_returns_it():
    """`set(obj, x)` on a parameter must mutate in place and hand back the same
    object, so an assignment `p = set(p, v)` never detaches `p` from its module."""
    p = pcx.Param(jnp.array(0.0))

    result = pcx.set(p, jnp.array(5.0))

    assert result is p
    assert_allclose(p.get(), 5.0)


def test_set_unwraps_a_parameter_source():
    """Assigning one parameter into another must copy the *value*, not nest a
    parameter inside a parameter — a nested Param would flatten to a leaf that is
    itself a pytree and corrupt the leaf count."""
    dst = pcx.Param(jnp.array(0.0))

    pcx.set(dst, pcx.Param(jnp.array(7.0)))

    assert not isinstance(dst.get(), pcx.BaseParam)
    assert_allclose(dst.get(), 7.0)


@pytest.mark.bug(
    "BUGS.md#12: pcx.set on a non-param calls set(x) with one argument: TypeError, missing required argument 'x'"
)
def test_set_on_a_plain_value_returns_the_new_value():
    """Documented contract: "otherwise return the new value itself".

    `set` exists so that call sites which do not know whether they hold a
    parameter or a bare value can write `obj = set(obj, new)` unconditionally.
    That is precisely the case this breaks on, so the helper cannot be used for
    the job it was written for.
    """
    assert pcx.set(1.0, 2.0) == 2.0


def test_paramdict_contributes_one_dynamic_leaf_per_entry():
    """A `ParamDict` is the cache behind `Vode`; its entries must be visible to
    jax as ordinary leaves or cached activations would not be differentiable."""
    pd = pcx.ParamDict({"a": jnp.ones(2), "b": jnp.zeros(3)})

    assert count_leaves(pd) == 2


def test_paramdict_indexing_returns_the_stored_value():
    pd = pcx.ParamDict({"a": jnp.ones(2)})

    assert_allclose(pd["a"], jnp.ones(2))
    assert "a" in pd


def test_paramdict_get_without_a_key_returns_the_whole_mapping():
    """`BaseParam.get()` takes no arguments, so `ParamDict.get()` has to keep
    working as the generic parameter accessor used by `tree_inject`."""
    contents = {"a": jnp.ones(2)}
    pd = pcx.ParamDict(contents)

    assert pd.get() is contents


def test_paramdict_get_returns_the_default_for_a_missing_key():
    pd = pcx.ParamDict({"a": jnp.ones(2)})

    assert pd.get("missing", "fallback") == "fallback"


def test_cleared_paramdict_contributes_no_dynamic_leaves():
    """`clear_params` sets a cache to None so the freed activations do not travel
    through the next transform. None is an empty pytree node, so the leaf count
    must drop to zero."""
    pd = pcx.ParamDict({"a": jnp.ones(2), "b": jnp.zeros(3)})

    pd.set(None)

    assert count_leaves(pd) == 0


def test_cleared_paramdict_accepts_new_entries():
    """A cleared cache has to be refillable; the next forward pass writes into
    exactly this state."""
    pd = pcx.ParamDict({"a": jnp.ones(2)})
    pd.set(None)

    pd["b"] = jnp.zeros(1)

    assert_allclose(pd["b"], jnp.zeros(1))


@pytest.mark.bug(
    "BUGS.md#7: ParamDict.__contains__ does not guard against a None value: TypeError, 'NoneType' is not iterable"
)
def test_membership_test_on_a_cleared_paramdict_is_false():
    """A cleared `ParamDict` is documented to be equivalent to an empty one —
    `__setitem__` says so explicitly by resetting `_value` to `{}`. Membership
    must therefore answer False rather than raise.

    This is the state `clear_params` leaves every cache in, and `Vode.energy`
    asks `"h" in cache` on exactly this object, so a whole inference step fails
    after a cache clear.
    """
    pd = pcx.ParamDict({"a": jnp.ones(2)})
    pd.set(None)

    assert ("a" in pd) is False


@pytest.mark.bug("BUGS.md#7: ParamDict.__getitem__ does not guard against a None value: TypeError, not subscriptable")
def test_indexing_a_cleared_paramdict_raises_key_error():
    """An empty mapping raises `KeyError` on a missing key — that is the error
    callers catch. A `TypeError` from `None` instead escapes any `except KeyError`
    written around a cache lookup and surfaces far from its cause."""
    pd = pcx.ParamDict({"a": jnp.ones(2)})
    pd.set(None)

    with pytest.raises(KeyError):
        pd["a"]


@pytest.mark.bug("BUGS.md#7: ParamDict.get does not guard against a None value: AttributeError on NoneType.get")
def test_get_on_a_cleared_paramdict_returns_the_default():
    """`get(key, default)` exists to be safe on absent keys; a cleared cache is
    the most absent a key can be, so it must yield the default rather than
    raise."""
    pd = pcx.ParamDict({"a": jnp.ones(2)})
    pd.set(None)

    assert pd.get("a", "fallback") == "fallback"


def test_param_survives_a_jit_boundary_as_an_argument():
    """The parameter has to be a legal jit argument — its treedef must be
    hashable and its value must arrive as a tracer — or nothing downstream in
    the library can work."""
    p = pcx.Param(jnp.array(2.0))

    out = jax.jit(lambda q: q.get() * 3.0)(p)

    assert_allclose(out, 6.0)
