"""`pcx.nn` layers against the bare equinox layers they wrap.

`pcx.nn.Layer` rebuilds an equinox module, replacing every array with a `LayerParam`
and everything else with a `StaticParam`. That split is the only thing standing between
a correct forward pass and a silently wrong one: misclassify a weight as static and it
stops receiving gradients; misclassify a shape or a flag as a parameter and it gets
"optimised" into nonsense. Neither shows up as an exception.

The oracle throughout is equinox itself, constructed with the *same* PRNG key, so the
two layers must agree bit for bit. The key is drawn from a private
`pcx.RandomKeyGenerator` rather than the global one, so these tests do not depend on
how many keys anything else has consumed.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from conftest import SEED, assert_allclose

import pcx
import pcx.nn as pxnn


def _key():
    """The first key of a fresh seed-``SEED`` stream — the same one the layer under
    test receives when handed an equally fresh generator."""
    return pcx.RandomKeyGenerator(seed=SEED)()


# Forward pass equivalence #############################################################


def test_linear_matches_bare_equinox_linear():
    """Identity: ``pxnn.Linear(a, b)(x) == eqx.nn.Linear(a, b, key=k)(x)`` for the same
    ``k``.

    This checks initialisation and the forward pass at once. If the wrapper mangled the
    weight/bias split, or dropped the bias into the static side, the outputs would
    differ by a constant that is easy to mistake for a different random init.
    """
    x = jnp.array([1.0, -2.0, 0.5])

    actual = pxnn.Linear(3, 2, rkg=pcx.RandomKeyGenerator(seed=SEED))(x)
    expected = eqx.nn.Linear(3, 2, True, key=_key())(x)

    assert jnp.array_equal(actual, expected), f"{actual} != {expected}"


def test_conv2d_matches_bare_equinox_conv2d():
    """Same identity for a convolution, whose static configuration (stride, padding,
    dilation, groups) is far richer than a Linear's. Any of those leaking into the
    dynamic side, or being lost, changes the output shape or the arithmetic.
    """
    x = jnp.arange(2 * 5 * 5, dtype=jnp.float32).reshape(2, 5, 5) / 10.0

    actual = pxnn.Conv2d(2, 3, 3, rkg=pcx.RandomKeyGenerator(seed=SEED))(x)
    expected = eqx.nn.Conv2d(2, 3, 3, key=_key())(x)

    assert actual.shape == expected.shape
    assert jnp.array_equal(actual, expected), f"max abs difference {jnp.max(jnp.abs(actual - expected))}"


def test_layernorm_matches_bare_equinox_layernorm():
    """LayerNorm has no randomness but does have both learnable arrays and a float
    ``eps`` that must stay static. Wrapping ``eps`` as a parameter would make it a
    gradient target and let it drift towards zero, which eventually divides by zero.
    """
    x = jnp.array([1.0, 2.0, 3.0, 10.0])

    actual = pxnn.LayerNorm((4,))(x)
    expected = eqx.nn.LayerNorm((4,))(x)

    assert jnp.array_equal(actual, expected), f"{actual} != {expected}"


def test_linear_bias_free_matches_bare_equinox():
    """``bias=False`` must reach equinox: a layer that keeps a bias the caller asked it
    to drop adds a learnable offset nobody accounts for.
    """
    x = jnp.array([1.0, -2.0, 0.5])

    actual = pxnn.Linear(3, 2, False, rkg=pcx.RandomKeyGenerator(seed=SEED))(x)
    expected = eqx.nn.Linear(3, 2, False, key=_key())(x)

    assert jnp.array_equal(actual, expected), f"{actual} != {expected}"


# Parameter / static split #############################################################


def test_layer_arrays_are_wrapped_as_layer_params():
    """Weights must be `LayerParam`s: that type is what ``pxu.M(pxnn.LayerParam)``
    selects, so a weight of any other type is invisible to the weight optimiser and
    never trains.
    """
    layer = pxnn.Linear(3, 2, rkg=pcx.RandomKeyGenerator(seed=SEED))

    assert isinstance(layer.nn.weight, pxnn.LayerParam)
    assert isinstance(layer.nn.bias, pxnn.LayerParam)


def test_non_array_attributes_are_not_dynamic_leaves():
    """A Linear has exactly two dynamic leaves — the weight and the bias.

    ``in_features``, ``out_features`` and ``use_bias`` describe the layer, they are not
    part of its state. If they were dynamic, jit would trace them, vmap would try to
    map over them, and the optimiser would add gradients to an integer.
    """
    layer = pxnn.Linear(3, 2, rkg=pcx.RandomKeyGenerator(seed=SEED))

    leaves = jtu.tree_leaves(layer)
    assert len(leaves) == 2, f"expected 2 dynamic leaves (weight, bias), got {len(leaves)}: {leaves}"
    assert {tuple(jnp.shape(leaf)) for leaf in leaves} == {(2, 3), (2,)}

    assert layer.nn.in_features == 3
    assert layer.nn.out_features == 2
    assert not isinstance(layer.nn.in_features, pcx.Param), "in_features was wrapped as a dynamic parameter"


# Weight sharing #######################################################################


def test_shared_layer_is_a_distinct_object_holding_the_same_weight():
    """`pxnn.shared` implements weight tying. The copy must be a different module — jax
    forbids two references to the same node in one pytree — while the weight object
    itself must be shared by identity, not merely equal by value.
    """
    layer = pxnn.Linear(3, 2, rkg=pcx.RandomKeyGenerator(seed=SEED))

    twin = pxnn.shared(layer)

    assert twin is not layer, "shared() returned the same module object"
    assert twin.nn is not layer.nn, "shared() returned the same inner equinox module"
    assert twin.nn.weight is layer.nn.weight, "shared() copied the weight instead of sharing it"
    assert twin.nn.bias is layer.nn.bias


def test_mutating_a_shared_weight_is_visible_through_both_layers():
    """The point of tying: one update, both layers. If the parameter were copied, a
    tied-weight autoencoder would quietly train two independent weight matrices and
    stop being tied at all.
    """
    layer = pxnn.Linear(3, 2, rkg=pcx.RandomKeyGenerator(seed=SEED))
    twin = pxnn.shared(layer)
    x = jnp.array([1.0, -2.0, 0.5])

    twin.nn.weight.set(jnp.zeros((2, 3)))

    assert jnp.array_equal(layer.nn.weight.get(), jnp.zeros((2, 3)))
    assert jnp.array_equal(layer(x), twin(x))


# Train / eval #########################################################################


def test_eval_makes_dropout_the_identity():
    """At evaluation time dropout must be exactly the identity — not "close to", not
    "rescaled". Any residual noise makes reported test accuracy irreproducible.
    """
    x = jnp.arange(8, dtype=jnp.float32) + 1.0
    dropout = pxnn.Dropout(0.5)

    dropout.eval()

    assert jnp.array_equal(dropout(x), x)
    assert jnp.array_equal(dropout(x, key=jax.random.PRNGKey(0)), x), "eval-mode dropout still consumed randomness"


def test_train_makes_dropout_stochastic_with_inverted_scaling():
    """In train mode dropout must actually drop, and must rescale the survivors by
    ``1 / (1 - p)`` so the expected activation is preserved.

    Both halves matter: a dropout that never drops is a silent no-op regulariser, and
    one that drops without rescaling shifts the mean of every downstream layer between
    training and evaluation.
    """
    p = 0.5
    x = jnp.ones(256)
    dropout = pxnn.Dropout(p)

    dropout.train()
    out = dropout(x, key=jax.random.PRNGKey(0))

    kept = 1.0 / (1.0 - p)
    assert bool(jnp.all((out == 0.0) | (out == kept))), "dropout produced values outside {0, 1/(1-p)}"
    assert bool(jnp.any(out == 0.0)), "train-mode dropout dropped nothing"
    assert bool(jnp.any(out == kept)), "train-mode dropout dropped everything"
    assert_allclose(jnp.mean(out), 1.0, rtol=0.15, atol=0.15)


def test_train_mode_dropout_requires_a_key():
    """Non-deterministic dropout without a key would have to invent randomness from
    somewhere, which is how a "reproducible" run stops being reproducible.
    """
    dropout = pxnn.Dropout(0.5)
    dropout.train()

    with pytest.raises(RuntimeError):
        dropout(jnp.ones(8))


def test_train_and_eval_set_the_module_mode_recursively():
    """`.train()`/`.eval()` are recursive: a nested layer left in the wrong mode is the
    classic source of an evaluation that quietly keeps dropping activations.
    """

    class Stack(pcx.Module):
        def __init__(self):
            super().__init__()
            self.dropout = pxnn.Dropout(0.5)
            self.linear = pxnn.Linear(3, 2, rkg=pcx.RandomKeyGenerator(seed=SEED))

    stack = Stack()

    stack.eval()
    assert stack.is_eval and stack.dropout.is_eval and stack.linear.is_eval

    stack.train()
    assert stack.is_train and stack.dropout.is_train and stack.linear.is_train
