"""Numerical correctness of `pcx.utils.Optim` and `pcx.utils.OptimTree`.

`Optim` is a thin wrapper around optax, and "thin" is the claim under test: a step
must move a parameter by exactly the amount the underlying update rule prescribes,
and by that amount only once. Every expectation below is either the hand-written
update rule (``w - lr * g``, or the momentum recurrence unrolled in Python) or optax
driven directly. Nothing is compared against a recorded pcx output.

Note that `Optim` takes a zero-argument *factory* (``lambda: optax.sgd(...)``), not an
optimizer instance, and that a masked parameter tree (``pxu.M(...)(model)``) must be
passed consistently to `init`, to `step`'s ``module`` and to its ``grads``.
"""

import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import pytest
from conftest import assert_allclose

import pcx
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
import pcx.utils as pxu


class _Params(pcx.BaseModule):
    """A bare parameter container.

    `pcx.BaseModule` (rather than `pcx.Module`) is used deliberately: it carries no
    train/eval state, so the pytree under test is exactly the parameters named here
    and the optimiser's filter has nothing else to trip over.
    """

    def __init__(self, param_type=pxnn.LayerParam, **params):
        for name, value in params.items():
            setattr(self, name, param_type(value))


def _grads_like(tree, value):
    """A gradient pytree matching ``tree``'s structure, every parameter set to ``value``.

    Built with `tree_map` so masked-out positions (``None``) stay ``None``, which is
    what `Optim.step` requires of a gradient tree.
    """
    return jtu.tree_map(
        lambda p: type(p)(jnp.full(jnp.shape(p.get()), value, dtype=jnp.float32)),
        tree,
        is_leaf=lambda x: isinstance(x, pcx.BaseParam),
    )


# Plain SGD ############################################################################


def test_sgd_step_is_exactly_weight_minus_learning_rate_times_gradient():
    """Update rule: ``w <- w - lr * g``. Nothing else.

    This is the most basic contract an optimiser has. If a wrapper rescales, clips or
    double-applies the update, every learning rate reported in a paper is wrong by
    that factor.
    """
    lr = 0.1
    w0 = jnp.array([1.0, 2.0, 3.0])
    b0 = jnp.array([0.0])
    g_w = jnp.array([1.0, -0.5, 4.0])
    g_b = jnp.array([2.0])

    model = _Params(w=w0, b=b0)
    grads = _Params(w=g_w, b=g_b)
    optim = pxu.Optim(lambda: optax.sgd(lr), model)

    optim.step(model, grads)

    assert jnp.array_equal(model.w.get(), w0 - lr * g_w), f"{model.w.get()} != {w0 - lr * g_w}"
    assert jnp.array_equal(model.b.get(), b0 - lr * g_b), f"{model.b.get()} != {b0 - lr * g_b}"


def test_sgd_with_momentum_follows_the_hand_computed_recurrence():
    """Momentum recurrence, unrolled by hand for three steps:

        m_t = mu * m_{t-1} + g_t,   w_t = w_{t-1} - lr * m_t,   m_0 = 0

    With a constant gradient the trace grows towards ``g / (1 - mu)``; an
    implementation that instead used ``m_t = mu * m_{t-1} + (1 - mu) * g_t`` would
    differ by exactly that factor and would look like a merely "slower" optimiser
    rather than a wrong one.
    """
    lr, mu = 0.1, 0.9
    w0 = jnp.array([1.0, -2.0])
    g = jnp.array([1.0, 0.5])

    model = _Params(w=w0)
    optim = pxu.Optim(lambda: optax.sgd(lr, momentum=mu), model)

    expected = w0
    trace = jnp.zeros_like(w0)
    for _ in range(3):
        trace = mu * trace + g
        expected = expected - lr * trace

        optim.step(model, _Params(w=g))

    assert_allclose(model.w.get(), expected)


def test_adam_step_matches_optax_driven_directly():
    """Cross-check against optax used as a plain function, with no pcx in the loop.

    Adam has state that pcx must thread through untouched; running the same
    transformation by hand on the same gradients is the strongest available oracle
    for a stateful optimiser.
    """
    w0 = jnp.array([1.0, -2.0, 0.5])
    grads = [jnp.array([1.0, 0.5, -1.0]), jnp.array([0.25, -0.5, 2.0])]

    reference_opt = optax.adam(1e-2)
    reference_w = w0
    reference_state = reference_opt.init(reference_w)
    for g in grads:
        updates, reference_state = reference_opt.update(g, reference_state, reference_w)
        reference_w = optax.apply_updates(reference_w, updates)

    model = _Params(w=w0)
    optim = pxu.Optim(lambda: optax.adam(1e-2), model)
    for g in grads:
        optim.step(model, _Params(w=g))

    assert_allclose(model.w.get(), reference_w)


# scale_by #############################################################################


@pytest.mark.bug("Optim.step(scale_by=k) writes g*k back into the caller's gradient Params in place")
def test_step_does_not_mutate_the_caller_gradients():
    """``scale_by`` is documented as scaling the gradients *before* handing them to
    optax — an input to the step, not a side effect on the caller's tree.

    Gradients are routinely reused: applied to two optimisers (weights and states), or
    inspected after the step. If ``step`` rewrites them in place, the second consumer
    silently sees ``k`` times the gradient it asked for.
    """
    g_value = 1.0
    grads = _Params(w=jnp.full((3,), g_value))
    model = _Params(w=jnp.zeros(3))
    optim = pxu.Optim(lambda: optax.sgd(1.0), model)

    optim.step(model, grads, scale_by=0.5)

    assert jnp.array_equal(grads.w.get(), jnp.full((3,), g_value)), (
        f"gradients were mutated: {grads.w.get()} != {jnp.full((3,), g_value)}"
    )


@pytest.mark.bug("Optim.step(scale_by=k) mutates grads in place, so repeated steps compound k")
def test_two_scaled_steps_apply_the_same_scaling_each_time():
    """Two SGD steps with the same gradients and the same ``scale_by`` must move the
    weight by ``2 * lr * k * g``.

    Identity: ``w_2 = w_0 - 2 * lr * k * g``. If the scaling is applied to a mutated
    gradient the second step sees ``k**2 * g``, so the trajectory bends away from the
    prescribed one after the very first reuse — a wrong number, not a crash.
    """
    lr, k, g_value = 1.0, 0.5, 1.0
    w0 = jnp.ones(2)

    model = _Params(w=w0)
    grads = _Params(w=jnp.full((2,), g_value))
    optim = pxu.Optim(lambda: optax.sgd(lr), model)

    optim.step(model, grads, scale_by=k)
    optim.step(model, grads, scale_by=k)

    expected = w0 - 2.0 * lr * k * g_value

    assert_allclose(model.w.get(), expected)


def test_a_single_scaled_step_scales_the_update():
    """``w_1 = w_0 - lr * k * g`` for one step: the scaling itself is applied correctly,
    which is what makes the compounding above a reuse bug rather than a scaling bug.
    """
    lr, k, g_value = 1.0, 0.5, 3.0
    w0 = jnp.ones(2)

    model = _Params(w=w0)
    optim = pxu.Optim(lambda: optax.sgd(lr), model)

    optim.step(model, _Params(w=jnp.full((2,), g_value)), scale_by=k)

    assert_allclose(model.w.get(), w0 - lr * k * g_value)


# Masking ##############################################################################


class _Mixed(pcx.BaseModule):
    """A weight and a value-node state, the two parameter families pcx optimises apart."""

    def __init__(self, weight, state):
        self.weight = pxnn.LayerParam(weight)
        self.state = pxc.VodeParam(state)


def test_masked_out_parameters_are_left_untouched():
    """An optimiser built on ``pxu.M(LayerParam)(model)`` must move weights and must not
    move value-node states.

    Predictive coding relies on this split: weights and states are updated by different
    optimisers on different schedules. A leak either way corrupts the algorithm rather
    than the loss curve.
    """
    lr, g_value = 1.0, 3.0
    weight0 = jnp.ones(2)
    state0 = jnp.ones(2)

    model = _Mixed(weight0, state0)
    masked = pxu.M(pxnn.LayerParam)(model)
    optim = pxu.Optim(lambda: optax.sgd(lr), masked)

    optim.step(masked, _grads_like(masked, g_value))

    assert_allclose(model.weight.get(), weight0 - lr * g_value)
    assert jnp.array_equal(model.state.get(), state0), "a masked-out parameter was updated"


def test_frozen_parameters_are_excluded_by_the_hasnot_mask():
    """``pxu.M_hasnot(VodeParam, frozen=True)`` is how the output node is pinned to the
    label during inference. A frozen parameter must be bit-identical after a step.
    """
    free0 = jnp.zeros(2)
    frozen0 = jnp.zeros(2)

    model = _Mixed(jnp.zeros(2), free0)
    model.frozen = pxc.VodeParam(frozen0)
    model.frozen.frozen = True

    masked = pxu.M_hasnot(pxc.VodeParam, frozen=True)(model)
    optim = pxu.Optim(lambda: optax.sgd(0.5), masked)

    optim.step(masked, _grads_like(masked, 1.0))

    assert jnp.array_equal(model.frozen.get(), frozen0), "a frozen parameter moved"
    assert_allclose(model.state.get(), free0 - 0.5)


# None gradients #######################################################################


def test_none_gradient_with_allow_none_leaves_the_optimiser_state_unchanged():
    """Skipping a step must be a true no-op for the state.

    With momentum, advancing the trace on a step whose update was never applied
    desynchronises the state from the parameters: the next real step then carries
    momentum from a gradient that was never used.
    """
    model = _Params(w=jnp.zeros(2))
    optim = pxu.Optim(lambda: optax.sgd(0.1, momentum=0.9), model)

    optim.step(model, _Params(w=jnp.ones(2)))
    state_before = jtu.tree_leaves(optim.state.get())
    w_before = model.w.get()

    skipped = _Params(w=jnp.ones(2))
    skipped.w.set(None)
    optim.step(model, skipped, allow_none=True)

    state_after = jtu.tree_leaves(optim.state.get())

    assert len(state_after) == len(state_before)
    for i, (before, after) in enumerate(zip(state_before, state_after, strict=True)):
        assert jnp.array_equal(before, after), f"optimiser state leaf {i} changed on a skipped step"
    assert jnp.array_equal(model.w.get(), w_before), "parameters moved on a skipped step"


def test_none_gradient_without_allow_none_raises():
    """The default is strict: a missing gradient is a bug in the caller's masking, and
    silently skipping it would hide a layer that never trains.
    """
    model = _Params(w=jnp.zeros(2))
    optim = pxu.Optim(lambda: optax.sgd(0.1), model)

    grads = _Params(w=jnp.ones(2))
    grads.w.set(None)

    with pytest.raises(ValueError):
        optim.step(model, grads)


# OptimTree ############################################################################


class _TwoStates(pcx.BaseModule):
    def __init__(self, a, b):
        self.a = pxc.VodeParam(a)
        self.b = pxc.VodeParam(b)


def test_optimtree_gives_each_leaf_an_independent_momentum_state():
    """`OptimTree` exists so that separately-scheduled parameter groups do not share
    optimiser state. With momentum this is observable: stepping only leaf ``a`` must
    leave leaf ``b``'s trace at zero.

    Recurrence for the stepped leaf: ``m_1 = g = 1``, ``m_2 = mu * m_1 + g = 1.9``,
    so ``a`` ends at ``-lr * (m_1 + m_2)``. A shared state would give ``b`` a non-zero
    trace and make its first real update depend on ``a``'s history.
    """
    lr, mu = 0.1, 0.9
    model = _TwoStates(jnp.zeros(2), jnp.zeros(2))
    optim = pxu.OptimTree(
        lambda: optax.sgd(lr, momentum=mu),
        lambda x: isinstance(x, pxc.VodeParam),
        model,
    )

    for _ in range(2):
        grads = _TwoStates(jnp.ones(2), jnp.ones(2))
        grads.b.set(None)
        optim.step(model, grads)

    optims = [
        leaf
        for leaf in jtu.tree_leaves(optim.state.get(), is_leaf=lambda x: isinstance(x, pxu.Optim))
        if isinstance(leaf, pxu.Optim)
    ]
    assert len(optims) == 2, f"expected one optimiser per leaf, got {len(optims)}"

    traces = [jtu.tree_leaves(o.state.get()) for o in optims]
    m1 = 1.0
    m2 = mu * m1 + 1.0

    assert_allclose(traces[0][0], jnp.full((2,), m2))
    assert jnp.array_equal(traces[1][0], jnp.zeros(2)), f"untouched leaf accumulated momentum: {traces[1][0]}"
    assert_allclose(model.a.get(), jnp.full((2,), -lr * (m1 + m2)))
    assert jnp.array_equal(model.b.get(), jnp.zeros(2)), "a leaf with no gradient was updated"
