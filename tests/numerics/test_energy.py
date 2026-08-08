"""Numerical correctness of the predictive-coding energy functions.

These are the scientific core of the library: every inference step and every weight
update is a derivative of one of these three functions. A constant factor or a sign
that is wrong here does not crash anything — the network still trains, just towards
something other than what the paper says. So every expected value below is written
out as a closed form, or produced by `jax.grad` of a hand-written function, or taken
from `optax`. Nothing is compared against what pcx happens to return.

Notation follows the predictive-coding literature: ``h`` is the value held by the
node, ``u`` the incoming prediction, and ``e = h - u`` the prediction error.
"""

import jax
import jax.numpy as jnp
import optax
import pytest
from conftest import assert_allclose

import pcx.predictive_coding as pxc


def _vode(h, u, energy_fn=pxc.se_energy):
    """A Vode holding ``h`` and ``u``, with no ruleset rewriting in the way.

    The default status is ``None``, so ``set("u", ...)`` matches no rule and simply
    stores ``u``; this is the plain, unbatched configuration the energies are defined on.
    """
    vode = pxc.Vode(energy_fn=energy_fn)
    vode.h.set(h)
    vode.set("u", u)

    return vode


# Squared error ########################################################################


def test_se_energy_equals_half_the_squared_prediction_error():
    """Identity: ``E(h, u) = 0.5 * (h - u)**2``, elementwise.

    The factor of one half is not cosmetic: it is what makes ``dE/dh`` equal the raw
    prediction error ``h - u``. Dropping it doubles every inference gradient, which
    silently doubles the effective inference rate of every experiment.
    """
    h = jnp.array([[1.0, -2.0], [0.5, 3.0]])
    u = jnp.array([[0.0, 1.0], [0.5, -1.5]])

    expected = 0.5 * (h - u) ** 2

    assert_allclose(pxc.se_energy(_vode(h, u)), expected)


def test_se_energy_is_never_negative():
    """A squared-error energy is a norm: it cannot be negative for any inputs.

    A negative energy would make the inference dynamics unbounded below, so the
    network could reduce its energy forever without ever matching the prediction.
    """
    key_h, key_u = jax.random.split(jax.random.PRNGKey(0))
    h = jax.random.normal(key_h, (7, 5)) * 10.0
    u = jax.random.normal(key_u, (7, 5)) * 10.0

    energy = pxc.se_energy(_vode(h, u))

    assert bool(jnp.all(energy >= 0.0)), f"negative energy: min {energy.min()}"


def test_se_energy_is_exactly_zero_when_the_prediction_is_perfect():
    """``E == 0`` exactly when ``h == u``, and strictly positive otherwise.

    Zero energy is the fixed point of inference. If the minimum sat anywhere other
    than ``h == u`` the network would converge to a biased state.
    """
    h = jnp.array([[1.0, -2.0, 0.0], [3.5, 7.0, -0.25]])

    assert bool(jnp.all(pxc.se_energy(_vode(h, h)) == 0.0))
    assert bool(jnp.all(pxc.se_energy(_vode(h, h + 1e-3)) > 0.0))


def test_se_energy_gradient_wrt_h_is_the_prediction_error():
    """THE defining identity of predictive coding: ``dE/dh == h - u``.

    Inference descends this gradient, so its sign decides whether a value node moves
    towards its prediction or away from it. A sign error here inverts inference while
    the loss curve still looks like it is training. Oracle is `jax.grad` applied to a
    function that only pcx's own energy is inside — the expectation ``h - u`` is written
    out by hand.
    """
    h = jnp.array([[1.0, -2.0], [0.5, 3.0]])
    u = jnp.array([[0.0, 1.0], [0.5, -1.5]])

    def total_energy(h_, u_):
        return pxc.se_energy(_vode(h_, u_)).sum()

    assert_allclose(jax.grad(total_energy, argnums=0)(h, u), h - u)


def test_se_energy_gradient_wrt_u_is_minus_the_prediction_error():
    """The other half of the identity: ``dE/du == -(h - u)``.

    This is the gradient that reaches the weights through the layer that produced
    ``u``. It must be exactly the negation of ``dE/dh``, otherwise the learning
    signal and the inference signal disagree about which way is downhill.
    """
    h = jnp.array([[1.0, -2.0], [0.5, 3.0]])
    u = jnp.array([[0.0, 1.0], [0.5, -1.5]])

    def total_energy(h_, u_):
        return pxc.se_energy(_vode(h_, u_)).sum()

    assert_allclose(jax.grad(total_energy, argnums=1)(h, u), -(h - u))


# Cross entropy ########################################################################


def test_ce_energy_matches_optax_softmax_cross_entropy():
    """Identity: ``sum_c E[c] == -sum_c h_c * log_softmax(u)_c``, i.e. optax's softmax
    cross entropy. pcx keeps the sum unreduced (one term per class), so the oracle is
    compared against the sum over the class axis.

    `optax.softmax_cross_entropy` is an independent, widely-tested implementation; if
    pcx disagrees with it, any classification result from the library is suspect.
    """
    logits = jnp.array([[2.0, -1.0, 0.5], [0.0, 0.0, 3.0]])
    targets = jax.nn.one_hot(jnp.array([0, 2]), 3)

    expected = optax.softmax_cross_entropy(logits=logits, labels=targets)
    actual = pxc.ce_energy(_vode(targets, logits, energy_fn=pxc.ce_energy)).sum(axis=-1)

    assert_allclose(actual, expected)


def test_ce_energy_gradient_wrt_u_is_softmax_minus_target():
    """Identity: for a one-hot ``h``, ``dE/du == softmax(u) - h``.

    This is the textbook softmax-cross-entropy gradient, and the whole reason the
    combination is used: the error signal is the difference between what the network
    believes and what it was told. Getting it wrong (for instance, forgetting that
    ``sum_c h_c == 1``) rescales every classification gradient.
    """
    logits = jnp.array([[2.0, -1.0, 0.5], [0.0, 0.0, 3.0]])
    targets = jax.nn.one_hot(jnp.array([0, 2]), 3)

    def total_energy(u_):
        return pxc.ce_energy(_vode(targets, u_, energy_fn=pxc.ce_energy)).sum()

    expected = jax.nn.softmax(logits, axis=-1) - targets

    assert_allclose(jax.grad(total_energy)(logits), expected)


def test_ce_energy_is_invariant_to_a_constant_shift_of_the_logits():
    """Softmax shift invariance: ``log_softmax(u + c) == log_softmax(u)``, so the
    energy must not change when a constant is added to every logit.

    A cross entropy that reacts to the shift is not normalising properly, which shows
    up as a spurious dependence on the scale of the preceding layer's bias.
    """
    logits = jnp.array([[2.0, -1.0, 0.5], [0.0, 0.0, 3.0]])
    targets = jax.nn.one_hot(jnp.array([0, 2]), 3)

    base = pxc.ce_energy(_vode(targets, logits, energy_fn=pxc.ce_energy))
    shifted = pxc.ce_energy(_vode(targets, logits + 7.25, energy_fn=pxc.ce_energy))

    assert_allclose(shifted, base, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("scale", [1e2, 1e4])
def test_ce_energy_stays_finite_for_extreme_logits(scale: float):
    """Numerical-stability guard: a naive ``-h * log(softmax(u))`` overflows and then
    produces NaN for large logits, whereas the ``log_softmax`` form does not.

    Confident, saturated logits are exactly what a trained classifier produces, so a
    NaN here would appear only late in training and poison the whole run.
    """
    logits = jnp.array([[scale, -scale, 0.0], [0.0, scale, scale / 2.0]])
    targets = jax.nn.one_hot(jnp.array([0, 1]), 3)

    energy = pxc.ce_energy(_vode(targets, logits, energy_fn=pxc.ce_energy))

    assert bool(jnp.all(jnp.isfinite(energy))), f"non-finite energy at scale {scale}: {energy}"


def test_ce_energy_is_zero_for_a_perfectly_confident_correct_prediction():
    """``-log softmax(u)_c -> 0`` as the correct logit dominates: the energy floor of a
    cross-entropy node is zero, reached only when the prediction is certain and right.
    """
    logits = jnp.array([[30.0, 0.0, 0.0]])
    targets = jnp.array([[1.0, 0.0, 0.0]])

    energy = pxc.ce_energy(_vode(targets, logits, energy_fn=pxc.ce_energy)).sum()

    assert_allclose(energy, 0.0, atol=1e-6)


# Zero energy ##########################################################################


def test_zero_energy_is_identically_zero():
    """``zero_energy`` unconstrains a node, so its value must be zero everywhere —
    any non-zero contribution would pull the node back towards a prior it is meant to
    be free of.
    """
    h = jnp.array([[1.0, -2.0], [0.5, 3.0], [7.0, 7.0]])
    u = jnp.zeros_like(h)

    assert bool(jnp.all(pxc.zero_energy(_vode(h, u, energy_fn=pxc.zero_energy)) == 0.0))


def test_zero_energy_gradient_wrt_h_is_zero():
    """An unconstrained node must receive no force from its own energy term:
    ``d/dh zero_energy == 0`` for every element of ``h``.
    """
    h = jnp.array([[1.0, -2.0], [0.5, 3.0]])
    u = jnp.zeros_like(h)

    def total_energy(h_):
        return pxc.zero_energy(_vode(h_, u, energy_fn=pxc.zero_energy)).sum()

    assert_allclose(jax.grad(total_energy)(h), jnp.zeros_like(h))


@pytest.mark.bug("zero_energy returns jnp.zeros((1,)) regardless of the node value's shape")
def test_zero_energy_has_the_same_shape_as_the_node_value():
    """``zero_energy`` is a drop-in replacement for ``se_energy``/``ce_energy``, both of
    which return one energy term per element of ``h``. It must therefore have the shape
    of ``h``.

    A fixed ``(1,)`` shape breaks the caller: ``Vode.energy`` reshapes the returned
    array to ``(batch, -1)``, which is impossible for a size-1 array whenever the batch
    size is not 1.
    """
    h = jnp.ones((3, 2))
    u = jnp.zeros((3, 2))

    energy = pxc.zero_energy(_vode(h, u, energy_fn=pxc.zero_energy))

    assert energy.shape == h.shape, f"expected shape {h.shape}, got {energy.shape}"
