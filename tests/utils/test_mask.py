"""The ``pxu.M`` filter DSL, read as a truth table over a mixed-parameter model.

``M`` is how a pcx program says *which* parameters an operation applies to: which ones
a gradient is taken with respect to, which ones an optimiser owns, which ones a vmap
maps over. It is a pure selection language, so every one of its properties is a set
identity that can be stated without running it — ``M(A)`` selects the ``A``-typed
params and nothing else, ``M(A) | M(B)`` selects the union, ``~M(A)`` selects the
complement, ``.to((a, b))`` sends unmatched params to ``a`` and matched ones to ``b``.

Those identities are asserted here against sets computed directly from the model with
``isinstance``, never against what the DSL happens to return. A selection bug is silent
by construction: too small a set means some parameters quietly stop being trained, too
large a set means parameters get updated by an optimiser that was never meant to touch
them. Neither raises.

The model under test deliberately mixes the three parameter families the library uses —
``LayerParam`` (weights), ``VodeParam``/``VodeParam.Cache`` (predictive-coding state) and
``StaticParam`` (configuration) — because the whole point of the DSL is telling them
apart.
"""

from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import pcx
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
import pcx.utils as pxu

# Helpers ##############################################################################


def _by_path(tree: Any) -> dict[str, Any]:
    """Every parameter position in ``tree``, keyed by its attribute path.

    ``None`` is an empty node in jax rather than a leaf, so a filtered-out parameter
    would simply vanish from ``tree_leaves``. Treating ``None`` as a leaf keeps the
    position visible, which is what lets a test distinguish "masked out" from "the mask
    changed the shape of the tree".
    """
    return {
        jtu.keystr(path): leaf
        for path, leaf in jtu.tree_leaves_with_path(tree, is_leaf=lambda x: x is None or isinstance(x, pcx.BaseParam))
    }


def _kept(tree: Any) -> dict[str, Any]:
    """The positions a mask kept, i.e. everything it did not replace with ``None``."""
    return {path: leaf for path, leaf in _by_path(tree).items() if leaf is not None}


def _paths_where(model: Any, predicate) -> set[str]:
    """The set of parameter paths satisfying ``predicate`` — the oracle for a selection.

    Computed from the model with plain ``isinstance``/``getattr``, so it is independent
    of the DSL it is used to check.
    """
    return {path for path, param in _by_path(model).items() if predicate(param)}


class _Model(pxc.EnergyModule):
    """A linear layer plus two Vodes, one of them frozen.

    That is the smallest model containing all of ``LayerParam``, ``VodeParam``,
    ``VodeParam.Cache`` and ``StaticParam``, and it mirrors the tutorials' convention of
    marking the output Vode with a custom ``frozen`` attribute.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin = pxnn.Linear(3, 2)
        self.free = pxc.Vode((2,))
        self.frozen = pxc.Vode((2,))


@pytest.fixture
def model() -> _Model:
    m = _Model()

    # Give every Vode a concrete value, so no selection can succeed or fail merely
    # because a param still holds ``None``.
    m.free(jnp.zeros((2,)))
    m.frozen(jnp.zeros((2,)))

    # `frozen` is not a library concept: the tutorials attach it themselves and then
    # select on it with `M_has`/`M_hasnot`. Attaching it here tests that same path.
    m.frozen.h.frozen = True

    return m


# Selection by type ####################################################################


def test_type_mask_selects_exactly_the_params_of_that_type(model):
    """``M(T)(model)`` keeps the ``T``-typed params, by identity, and nothing else.

    This is the whole contract in one line. Keeping too few params silently freezes
    weights; keeping too many hands an optimiser parameters it was never given a state
    for. The identity check matters as well: the mask is a *selection*, so the surviving
    leaves must be the model's own ``Param`` objects, not copies — an optimiser step
    writes through them.
    """
    masked = pxu.M(pxnn.LayerParam)(model)

    expected = _paths_where(model, lambda p: isinstance(p, pxnn.LayerParam))
    assert expected == {".lin.nn.weight", ".lin.nn.bias"}, "fixture no longer has the parameters this test assumes"
    assert _kept(masked).keys() == expected

    for path, param in _kept(masked).items():
        assert param is _by_path(model)[path], f"{path} was copied rather than selected"


def test_unselected_params_become_none_without_changing_the_tree_shape(model):
    """Everything the mask rejects is replaced by ``None`` *in place*.

    The masked tree is used as a prefix pytree by the transforms (a gradient mask, a
    vmap axis spec). If rejection dropped a position instead of nulling it, the mask
    would no longer line up with the model it describes, and jax would either raise a
    structure error or, worse, align the wrong parameter with the wrong axis.
    """
    masked = pxu.M(pxc.VodeParam)(model)

    assert _by_path(masked).keys() == _by_path(model).keys()

    rejected = _by_path(model).keys() - _paths_where(model, lambda p: isinstance(p, pxc.VodeParam))
    assert all(_by_path(masked)[path] is None for path in rejected)


def test_none_mask_selects_every_param(model):
    """``M(None)`` matches unconditionally — the "select everything" spelling.

    The tutorials build an all-``False`` vmap spec with ``M(None).to([False, False])``,
    which only works if a ``None`` mask is a tautology rather than a mask on the type
    ``None``.
    """
    assert _kept(pxu.M(None)(model)).keys() == _by_path(model).keys()


def test_callable_mask_selects_by_predicate(model):
    """A plain callable is a valid mask, and means exactly the predicate it computes.

    This is the DSL's escape hatch: anything not expressible as a type or an attribute
    test goes through here, so it must agree with the type spelling where the two
    overlap.
    """
    by_callable = _kept(pxu.M(lambda p: isinstance(p, pxnn.LayerParam))(model))
    by_type = _kept(pxu.M(pxnn.LayerParam)(model))

    assert by_callable.keys() == by_type.keys()


# Set algebra ##########################################################################


def test_union_type_selects_the_union_of_the_two_type_masks(model):
    """``M(A | B)`` is the union of ``M(A)`` and ``M(B)``.

    ``VodeParam | VodeParam.Cache`` is the spelling every tutorial uses to batch the
    predictive-coding state. If the union collapsed to one branch, half of that state
    would be left unbatched and silently broadcast across the batch.
    """
    union = _kept(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache)(model)).keys()

    assert union == _kept(pxu.M(pxc.VodeParam)(model)).keys() | _kept(pxu.M(pxc.VodeParam.Cache)(model)).keys()
    assert union == _paths_where(model, lambda p: isinstance(p, pxc.VodeParam | pxc.VodeParam.Cache))


def test_or_between_two_masks_selects_the_union(model):
    """``M(A) | M(B)`` means union, like the type-level ``|`` it exists to generalise.

    The operator is the only way to combine two *masks* (as opposed to two types), so
    an optimiser told to own "weights or unfrozen states" depends on it being a real
    disjunction.
    """
    combined = _kept((pxu.M(pxnn.LayerParam) | pxu.M(pxc.VodeParam))(model)).keys()

    assert combined == _paths_where(model, lambda p: isinstance(p, pxnn.LayerParam | pxc.VodeParam))


def test_and_between_two_masks_selects_the_intersection(model):
    """``M(A) & M(B)`` means intersection, and an intersection of disjoint types is empty.

    An ``&`` that behaved like ``|`` would be invisible whenever one operand is a
    superset of the other, which is the common case (``Param`` and ``VodeParam``), and
    would quietly over-select everywhere else.
    """
    narrowed = _kept((pxu.M(pcx.Param) & pxu.M(pxc.VodeParam))(model)).keys()
    assert narrowed == _paths_where(model, lambda p: isinstance(p, pxc.VodeParam))

    disjoint = _kept((pxu.M(pxnn.LayerParam) & pxu.M(pxc.VodeParam))(model))
    assert disjoint == {}, "no parameter is both a LayerParam and a VodeParam"


def test_m_is_is_equivalent_to_chaining_and(model):
    """``M_is(a, b)`` is documented as ``M(a) & M(b)``; the two must agree.

    It is the public spelling of the conjunction, so a divergence would mean the
    documented equivalence in its own docstring is false.
    """
    assert (
        _kept(pxu.M_is(pcx.Param, pxc.VodeParam)(model)).keys()
        == _kept((pxu.M(pcx.Param) & pxu.M(pxc.VodeParam))(model)).keys()
    )


# Negation #############################################################################


def test_negation_of_a_leaf_predicate_is_the_logical_not(model):
    """``(~M(A)).apply(p)`` must be ``not M(A).apply(p)`` for every parameter.

    ``apply`` is the per-leaf predicate the whole DSL is built out of: every combinator
    resolves its operands through it. If negation is a no-op here, then ``~`` is a no-op
    everywhere it is composed, and the resulting selections are wrong without any
    diagnostic.
    """
    positive = pxu.M(pxnn.LayerParam)
    negative = ~positive

    for path, param in _by_path(model).items():
        assert negative.apply(param) == (not positive.apply(param)), f"~ did not negate at {path}"


def test_negation_applied_to_a_model_selects_the_complement(model):
    """``(~M(A))(model)`` must select every parameter ``M(A)`` rejects, and no other.

    This is the ordinary user-facing use — "optimise everything except the weights".
    ``M.__call__`` is the tree-application entry point and ``apply`` is the per-leaf
    predicate, so negation belongs on ``apply``. Installed on ``__call__`` instead it
    returns a scalar, and the caller gets something unusable as a mask.
    """
    masked = (~pxu.M(pxnn.LayerParam))(model)

    complement = _by_path(model).keys() - _paths_where(model, lambda p: isinstance(p, pxnn.LayerParam))
    assert _kept(masked).keys() == complement


def test_and_with_a_negated_mask_excludes_the_negated_set(model):
    """``M(A) & ~M(B)`` selects the ``A``s that are not ``B``s.

    This composed form is the DSL's own documented example
    (``(M(A | B)) & ~M_has(C, attr1=1)``). The combinators resolve their operands
    through ``apply``, so a ``_M_not`` that does not override ``apply`` leaves them
    resolving the *positive* predicate: the expression silently computes
    ``M(A) & M(B)`` — here an empty selection where the whole layer was meant to be
    chosen.
    """
    masked = (pxu.M(pcx.Param) & ~pxu.M(pxc.VodeParam))(model)

    expected = _paths_where(model, lambda p: isinstance(p, pcx.Param) and not isinstance(p, pxc.VodeParam))
    assert expected == {".lin.nn.weight", ".lin.nn.bias"}, "fixture no longer has the parameters this test assumes"
    assert _kept(masked).keys() == expected


def test_de_morgan_holds_for_the_leaf_predicate(model):
    """``~(M(A) | M(B))`` and ``~M(A) & ~M(B)`` must agree, and both must be the complement.

    De Morgan is the sanity check that ``~``, ``&`` and ``|`` are one algebra rather
    than three unrelated behaviours. A user who refactors a filter from one form to the
    other — an entirely value-preserving edit in every other language — must get the
    same set back.
    """
    a, b = pxu.M(pxnn.LayerParam), pxu.M(pxc.VodeParam)

    for path, param in _by_path(model).items():
        expected = not (isinstance(param, pxnn.LayerParam) or isinstance(param, pxc.VodeParam))
        assert (~(a | b)).apply(param) == expected, f"~(A | B) wrong at {path}"
        assert (~a & ~b).apply(param) == expected, f"~A & ~B wrong at {path}"


@pytest.mark.bug("#88: `M.__call__` reffs the tree first, so a negated mask selects the `_BaseParamRef` placeholders")
def test_negation_does_not_select_deduplication_placeholders():
    """A mask must select parameters, never the bookkeeping `tree_ref` leaves behind.

    `M.__call__` calls `tree_ref` first, which replaces every duplicate `BaseParam`
    reference with a `_BaseParamRef` holding an integer index. A positive mask rejects
    those because they are not the requested type; a negated mask selects them for the
    same reason. They then reach `Optim.apply_updates`, whose `get(p)` returns the raw
    index and adds it to the parameter, so a shared model trains on numbers that are
    off by the ref index with nothing raised.
    """

    class Shared(pcx.Module):
        def __init__(self):
            super().__init__()
            self.a = pxnn.Linear(2, 2)
            self.b = self.a

    selected = _kept((~pxu.M(pxc.VodeParam))(Shared())).values()

    assert not [p for p in selected if type(p).__name__ == "_BaseParamRef"], (
        "a negated mask selected the de-duplication placeholders inserted by tree_ref"
    )


# Mapping to values ####################################################################


def test_to_maps_unmatched_to_the_first_element_and_matched_to_the_second(model):
    """``.to((a, b))`` sends rejected params to ``a`` and selected params to ``b``.

    The pair is indexed by the boolean result of the filter, so the order is trivially
    easy to invert — and an inversion is close to undetectable. ``.to((None, 0))`` is
    the tutorials' vmap spec: inverted, every weight would be treated as batched along
    axis 0 and every activation as shared, which changes the arithmetic without
    changing any shape the user sees.
    """
    selected = _paths_where(model, lambda p: isinstance(p, pxnn.LayerParam))
    rejected = _by_path(model).keys() - selected

    axes = _by_path(pxu.M(pxnn.LayerParam).to((None, 0))(model))
    assert all(axes[path] == 0 for path in selected), "selected params must map to the second element"
    assert all(axes[path] is None for path in rejected), "rejected params must map to the first element"

    flags = _by_path(pxu.M(pxnn.LayerParam).to((False, True))(model))
    assert {path for path, flag in flags.items() if flag is True} == selected
    assert {path for path, flag in flags.items() if flag is False} == rejected


def test_to_mutates_the_mask_and_the_mapping_persists_across_calls(model):
    """``.to`` is a setter on a stateful object, not a way of deriving a new mask.

    It returns ``self``, so ``M(T).to(...)`` reads like a builder — but the mutation is
    to the shared instance. A mask bound to a name and reused (the natural way to avoid
    repeating a filter) starts returning mapped values at every earlier call site once
    anyone calls ``.to`` on it. Pinned here so the footgun is visible rather than
    discovered in a training loop.
    """
    mask = pxu.M(pxnn.LayerParam)
    assert isinstance(next(iter(_kept(mask(model)).values())), pxnn.LayerParam)

    assert mask.to((False, True)) is mask, "`.to` is documented to return the mask itself"
    assert all(isinstance(v, bool) for v in _by_path(mask(model)).values()), (
        "a later call through the same instance is silently changed by `.to`"
    )

    assert mask.to(None) is mask
    assert isinstance(next(iter(_kept(mask(model)).values())), pxnn.LayerParam)


# Selection by attribute ###############################################################


def test_m_has_selects_the_params_carrying_the_attribute_value(model):
    """``M_has(T, attr=v)`` selects the ``T``s whose ``attr`` equals ``v`` — only those.

    The ``frozen=True`` convention is how every tutorial keeps the clamped output node
    out of the inference optimiser. Selecting the frozen node would let inference move
    the target it is supposed to be pinned to, which converges to nothing meaningful.
    """
    masked = pxu.M_has(pxc.VodeParam, frozen=True)(model)

    assert _kept(masked).keys() == {".frozen.h"}
    assert _kept(masked)[".frozen.h"] is model.frozen.h


def test_m_hasnot_selects_the_params_of_the_type_without_that_attribute_value(model):
    """``M_hasnot(T, attr=v)`` selects the ``T``s *not* carrying ``attr == v``.

    Note it stays inside ``T``: it is the type mask conjoined with the negated attribute
    test, not the complement of ``M_has``. Params of other types are excluded, which is
    exactly what ``optim_h.init(M_hasnot(VodeParam, frozen=True)(model))`` relies on —
    the weights must not end up in the activation optimiser.
    """
    masked = pxu.M_hasnot(pxc.VodeParam, frozen=True)(model)

    assert _kept(masked).keys() == {".free.h"}
    assert _kept(masked)[".free.h"] is model.free.h


def test_has_and_hasnot_methods_agree_with_the_module_level_helpers(model):
    """``M(T).has(**a)`` is documented as equal to ``M_has(T, **a)``; likewise ``hasnot``.

    Two spellings of one filter that disagreed would make the docstring's stated
    equivalence a trap, and both appear in user code.
    """
    assert (
        _kept(pxu.M(pxc.VodeParam).has(frozen=True)(model)).keys()
        == _kept(pxu.M_has(pxc.VodeParam, frozen=True)(model)).keys()
    )
    assert (
        _kept(pxu.M(pxc.VodeParam).hasnot(frozen=True)(model)).keys()
        == _kept(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model)).keys()
    )


def test_m_has_also_matches_attributes_forwarded_from_the_wrapped_array(model):
    """Attribute filters see the array's attributes too, because ``Param`` forwards them.

    ``Param.__getattr__`` delegates to the wrapped value, so ``hasattr(param, "shape")``
    is true even though no one set ``shape`` on the param. Attribute names therefore
    live in the same namespace as every ndarray attribute, and a user-chosen marker that
    happens to collide (``size``, ``dtype``, ``shape``, ``T``) silently filters on the
    array instead. Pinned so the collision is documented behaviour rather than a
    surprise.
    """
    masked = pxu.M_has(None, shape=(2, 3))(model)

    assert _kept(masked).keys() == {".lin.nn.weight"}
    assert _kept(masked)[".lin.nn.weight"].get().shape == (2, 3)


def test_m_has_ignores_params_lacking_the_attribute_entirely(model):
    """A param without the attribute is rejected, not treated as a match on ``None``.

    ``hasattr`` is checked before ``getattr``, so the ordinary case — most params never
    heard of ``frozen`` — must be a clean non-match rather than an ``AttributeError``
    escaping from inside a ``tree_map``.
    """
    assert _kept(pxu.M_has(None, frozen=True)(model)).keys() == {".frozen.h"}
    assert _kept(pxu.M_has(None, definitely_not_an_attribute=1)(model)) == {}


# Shared parameters ####################################################################


class _Shared(pcx.Module):
    """One parameter reachable through two attributes — a pydag rather than a pytree."""

    def __init__(self) -> None:
        super().__init__()
        p = pxnn.LayerParam(jnp.ones((2,)))
        self.a = p
        self.b = p


def test_a_shared_param_is_selected_only_once():
    """A param reachable twice appears in the mask once, as ``M.__call__`` documents.

    The mask reffs its input precisely so that "the mask will not have duplicates". A
    duplicated selection would hand the same array to an optimiser twice and apply the
    update twice per step.
    """
    shared = _Shared()

    masked = pxu.M(pxnn.LayerParam)(shared)

    assert _kept(masked).keys() == {".a"}
    assert _kept(masked)[".a"] is shared.a


def test_is_pytree_true_matches_the_default_path_on_an_already_reffed_tree():
    """``is_pytree=True`` only skips the reffing step; the selection must be unchanged.

    The flag exists as a performance shortcut for callers that already hold a reffed
    tree. If it changed the result, every transform that pre-reffs its arguments would
    compute a different mask from the same expression.
    """
    shared = _Shared()

    default = _by_path(pxu.M(pxnn.LayerParam).to((False, True))(shared))
    preffed = _by_path(pxu.M(pxnn.LayerParam).to((False, True))(pcx.tree_ref(shared), is_pytree=True))

    assert default == preffed
