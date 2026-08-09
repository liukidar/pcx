"""``pxnn.StatefulLayer`` against the equinox stateful layers it wraps.

Equinox keeps layer state (BatchNorm's running mean and variance, say) outside the
module: ``eqx.nn.make_with_state`` returns a ``(layer, state)`` pair, and every call
threads the state through explicitly. ``StatefulLayer`` hides that thread — it stores
the state in a single ``StateParam`` and re-injects it on each call — so the wrapper
owns two things that can silently go wrong:

* the **split**, deciding which leaves are trainable ``LayerParam``s, which are
  ``StaticParam`` configuration and which belong to the state. Running statistics that
  end up on the trainable side get gradient updates; weights that end up static stop
  learning. Neither raises.
* the **threading**, feeding the stored state in and writing the returned state back.
  Drop the write-back and the running statistics stay at their initial values forever,
  which only shows up as bad eval-time numbers, long after the fact.

The oracle throughout is equinox driven by hand with the same arguments and the same
input: the wrapper is a convenience, so it must be bit-for-bit indistinguishable from
the thing it wraps. Where a closed form exists (the eval-mode normalisation) the
expected value is written out as arithmetic instead.

``pxf.vmap`` cannot run on this jax version (BUGS.md #4), so the BatchNorm tests use
``jax.vmap`` directly with ``axis_name="batch"`` — the pattern equinox's own
documentation prescribes — and lift the updated state out through ``out_axes=None``.
"""

import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from conftest import SEED, assert_allclose, count_leaves

import pcx
import pcx.nn as pxnn
import pcx.utils as pxu

# `mode="ema"` reached eqx.nn.BatchNorm after 0.11.x, and equinox is version-locked to
# jax, so on an older jax the whole file is untestable. Gate on the capability rather
# than on a version number: this is about what the installed equinox can do, and a
# version comparison would go stale the moment the API moves again.
_BATCHNORM_HAS_MODE = "mode" in inspect.signature(eqx.nn.BatchNorm.__init__).parameters
pytestmark = pytest.mark.skipif(
    not _BATCHNORM_HAS_MODE,
    reason="eqx.nn.BatchNorm predates the `mode` argument; upgrade equinox to exercise StatefulLayer",
)

# A batch whose per-channel statistics are easy to write down by hand.
X = jnp.arange(12.0).reshape(4, 3)

# Non-default settings, so an argument passed through in the wrong position shows up.
EPS = 0.25
MOMENTUM = 0.75


# Test doubles #########################################################################


class _Counter(eqx.Module):
    """A minimal stateful equinox layer: ``y = x * weight + (number of prior calls)``.

    Deterministic, needs no vmap and no PRNG, so it isolates the wrapper's state
    threading from anything BatchNorm does. The count is readable straight out of the
    output, which makes "the state was not written back" a visible wrong number rather
    than an invisible one.
    """

    index: eqx.nn.StateIndex
    weight: jax.Array

    def __init__(self, weight: float) -> None:
        self.weight = jnp.asarray(weight)
        self.index = eqx.nn.StateIndex(jnp.array(0.0))

    def __call__(self, x, state, *, key=None):
        count = state.get(self.index)
        return x * self.weight + count, state.set(self.index, count + 1.0)


class _KeyEcho(eqx.Module):
    """A stateful layer that stores whatever key it was called with, and returns ``x``."""

    index: eqx.nn.StateIndex

    def __init__(self) -> None:
        self.index = eqx.nn.StateIndex(jnp.zeros((2,), dtype=jnp.uint32))

    def __call__(self, x, state, *, key=None):
        return x, state.set(self.index, jnp.asarray(key))


# Helpers ##############################################################################


def _params(module) -> list:
    """Every ``BaseParam`` in a module, in flatten order."""
    return list(jtu.tree_leaves(module, is_leaf=lambda x: isinstance(x, pcx.BaseParam)))


def _pcx_batchnorm(**kwargs) -> pxnn.BatchNorm:
    return pxnn.BatchNorm(3, "batch", EPS, True, MOMENTUM, mode="ema", **kwargs)


def _eqx_batchnorm():
    """The same layer built through the bare equinox API — the oracle."""
    return eqx.nn.make_with_state(eqx.nn.BatchNorm)(
        3, "batch", eps=EPS, channelwise_affine=True, momentum=MOMENTUM, mode="ema"
    )


def _run_pcx(layer, x):
    """One batched forward pass, returning the output and the updated state.

    The state is lifted out with ``out_axes=None`` (equinox's documented BatchNorm
    idiom) and written back explicitly, because the write ``StatefulLayer`` performs
    inside the trace leaves a batch tracer in the param.
    """

    def forward(xi):
        return layer(xi), layer.state.get()

    out, state = jax.vmap(forward, axis_name="batch", out_axes=(0, None))(x)
    layer.state.set(state)

    return out, state


def _run_eqx(layer, state, x):
    return jax.vmap(layer, axis_name="batch", in_axes=(0, None), out_axes=(0, None))(x, state)


# The parameter split ##################################################################


def test_weights_become_individual_params_and_the_state_a_single_state_param():
    """Each array of the wrapped layer gets its own ``LayerParam``; the whole state gets one.

    The asymmetry is deliberate and documented in the source: collapsing the state into
    a single ``StateParam`` is what makes it selectable as a unit. Splitting it per array
    instead would change which leaves a mask picks up; merging the weights would make the
    layer's parameters un-addressable individually.
    """
    layer = _pcx_batchnorm()

    layer_params = [p for p in _params(layer) if isinstance(p, pxnn.LayerParam)]
    state_params = [p for p in _params(layer) if isinstance(p, pxnn.StateParam)]

    assert len(layer_params) == 2, "BatchNorm's weight and bias must each be a LayerParam"
    assert {p.get().shape for p in layer_params} == {(3,)}
    assert len(state_params) == 1, "the whole state must collapse into exactly one StateParam"

    # ...and that one param really does hold the several arrays of the equinox state.
    assert count_leaves(state_params[0]) == 3


def test_the_state_is_not_selected_by_a_layer_param_mask():
    """``M(LayerParam)`` must not reach the running statistics.

    Every training loop optimises ``M(LayerParam)(model)``. Running statistics are
    accumulated averages, not gradient targets: if the mask picked them up, the
    optimiser would apply weight decay and momentum to them and the eval-time
    normalisation would drift away from the data it is supposed to describe.
    """
    layer = _pcx_batchnorm()

    selected = jtu.tree_leaves(pxu.M(pxnn.LayerParam)(layer), is_leaf=lambda x: isinstance(x, pcx.BaseParam))

    assert all(isinstance(p, pxnn.LayerParam) for p in selected)
    assert not any(isinstance(p, pxnn.StateParam) for p in selected)
    assert len(selected) == 2


def test_only_arrays_are_dynamic_leaves():
    """Configuration stays static: the layer flattens to arrays and nothing else.

    ``axis_name`` is a string and ``inference`` a bool, and both are dynamic leaves of
    the bare equinox module. If the wrapper left them dynamic, every jax transform over
    the layer would try to trace a string.
    """
    layer = _pcx_batchnorm()

    leaves = jtu.tree_leaves(layer)

    assert len(leaves) == 5, "weight, bias, and the three state arrays"
    assert all(isinstance(leaf, jax.Array) for leaf in leaves), [type(leaf).__name__ for leaf in leaves]


def test_the_filter_argument_decides_which_leaves_are_trainable():
    """``filter`` selects the trainable leaves; everything it rejects becomes static.

    It defaults to "is an array", but a caller passing their own predicate is the
    documented way to freeze part of a wrapped layer. A filter that were ignored would
    hand back a fully trainable layer while the caller believed otherwise.
    """
    trainable = pxnn.StatefulLayer(_Counter, 2.0)
    frozen = pxnn.StatefulLayer(_Counter, 2.0, filter=lambda w: False)

    assert any(isinstance(p, pxnn.LayerParam) for p in _params(trainable))
    assert not any(isinstance(p, pxnn.LayerParam) for p in _params(frozen))

    # A rejected leaf becomes a StaticParam, which by definition contributes nothing to
    # the dynamic side: the weight disappears from the flattened layer entirely.
    assert count_leaves(trainable) == 2, "the weight and the state array"
    assert count_leaves(frozen) == 1, "only the state array is left dynamic"

    # The layer must still compute the same thing either way: freezing is about which
    # params an optimiser can see, not about the forward pass.
    assert trainable(jnp.array(3.0)) == frozen(jnp.array(3.0))


# The oracle ###########################################################################


def test_wrapped_batchnorm_matches_bare_equinox_batchnorm():
    """A wrapped BatchNorm must equal equinox's, output and state, call after call.

    This is the only test that can catch a wrapping error as such — a mis-ordered
    positional argument (``eps`` and ``momentum`` are adjacent floats), a state fed in
    stale, a state written back unchanged. Two successive calls are needed because the
    first one initialises the running statistics to the batch statistics; only the
    second exercises the momentum blend, and so only the second would notice ``eps`` and
    ``momentum`` swapped.
    """
    layer = _pcx_batchnorm()
    reference, ref_state = _eqx_batchnorm()

    for call in range(2):
        out, state = _run_pcx(layer, X)
        expected_out, ref_state = _run_eqx(reference, ref_state, X)

        assert_allclose(out, expected_out, err_msg=f"output differs on call {call}")
        for actual, desired in zip(jtu.tree_leaves(state), jtu.tree_leaves(ref_state), strict=True):
            assert_allclose(actual, desired, err_msg=f"state differs on call {call}")


# State evolution ######################################################################


def test_running_statistics_change_between_training_calls():
    """In train mode the stored state must actually move.

    If the wrapper fed the state in but never wrote the update back, the forward pass
    would still be correct — BatchNorm normalises by the *batch* statistics during
    training — and the only symptom would appear at eval time, where the layer would
    normalise by the initial zeros. This is the assertion that separates the two.
    """
    layer = _pcx_batchnorm()

    before = jtu.tree_leaves(layer.state.get())
    _run_pcx(layer, X)
    after = jtu.tree_leaves(layer.state.get())

    assert any(not jnp.array_equal(a, b) for a, b in zip(before, after, strict=True))

    # The first training call has no history to blend with, so the running statistics
    # are exactly the batch statistics.
    mean, var = jnp.mean(X, axis=0), jnp.var(X, axis=0)
    assert_allclose(after[1], mean)
    assert_allclose(after[2], var)


def test_the_state_advances_once_per_call():
    """Every call reads the state the previous call wrote — no more, no less.

    A wrapper that re-injected the *initial* state each time, or that applied the update
    twice, produces perfectly plausible numbers. Counting calls in the state makes the
    off-by-one visible.
    """
    layer = pxnn.StatefulLayer(_Counter, 2.0)

    assert [layer(jnp.array(1.0)) for _ in range(3)] == [2.0, 3.0, 4.0]


def test_the_forward_key_reaches_the_wrapped_layer():
    """``key=`` is forwarded to the equinox call rather than swallowed.

    Stateful layers with randomness (dropout-like behaviour, stochastic normalisation)
    take their key per call. A key silently replaced by ``None`` makes such a layer
    either raise deep inside equinox or, worse, fall back to a fixed stream.
    """
    layer = pxnn.StatefulLayer(_KeyEcho)
    key = jax.random.PRNGKey(SEED)

    layer(jnp.array(1.0), key=key)

    assert jnp.array_equal(layer.state.get().get(layer.nn.index), key)


# Train / eval #########################################################################


def test_eval_mode_freezes_the_running_statistics():
    """``.eval()`` must stop the state from moving.

    Validation runs through the same code path as training. If evaluating a model
    updated its running statistics, the validation set would leak into the statistics
    used to report on it, and the reported number would depend on how many times it had
    been computed.
    """
    layer = _pcx_batchnorm()
    _run_pcx(layer, X)

    layer.eval()
    before = jtu.tree_leaves(layer.state.get())
    layer(X[0])
    after = jtu.tree_leaves(layer.state.get())

    assert all(jnp.array_equal(a, b) for a, b in zip(before, after, strict=True))


def test_eval_mode_normalises_with_the_stored_statistics():
    """In eval the output is ``(x - running_mean) / sqrt(running_var + eps)``.

    The closed form is the point of keeping running statistics at all: eval must be a
    per-sample function, independent of whatever else shares the batch. A layer that
    kept using batch statistics in eval would give a different answer for the same input
    depending on its neighbours — non-reproducible and, for a batch of one, degenerate.
    """
    layer = _pcx_batchnorm()
    _run_pcx(layer, X)
    layer.eval()

    mean, var = jnp.mean(X, axis=0), jnp.var(X, axis=0)
    x = jnp.array([1.0, 2.0, 3.0])

    # weight and bias are initialised to 1 and 0, so the affine part is the identity.
    assert_allclose(layer(x), (x - mean) / jnp.sqrt(var + EPS))


def test_train_mode_resumes_updating_after_eval():
    """``.train()`` undoes ``.eval()``; the inference flag is not a one-way switch.

    A model is evaluated between epochs and then trained again. If ``.eval()`` stuck,
    the running statistics would silently freeze at their epoch-one values for the rest
    of the run.
    """
    layer = _pcx_batchnorm()
    _run_pcx(layer, X)

    layer.eval()
    layer.train()

    before = jtu.tree_leaves(layer.state.get())
    _run_pcx(layer, X + 100.0)
    after = jtu.tree_leaves(layer.state.get())

    assert any(not jnp.array_equal(a, b) for a, b in zip(before, after, strict=True))


# Pytree behaviour #####################################################################


def test_the_state_survives_a_flatten_unflatten_round_trip():
    """Flattening and rebuilding a layer preserves its state exactly.

    Every jax transform round-trips its arguments through the pytree registry. A state
    lost or reset in the process would mean a jitted training step silently starts from
    the initial statistics on every call.
    """
    layer = pxnn.StatefulLayer(_Counter, 2.0)
    for _ in range(3):
        layer(jnp.array(1.0))

    leaves, treedef = jtu.tree_flatten(layer)
    rebuilt = jtu.tree_unflatten(treedef, leaves)

    assert rebuilt(jnp.array(1.0)) == layer(jnp.array(1.0)) == 5.0


def test_the_round_trip_produces_an_independent_state():
    """The rebuilt layer owns its state; mutating one must not move the other.

    ``StatefulLayer.__call__`` mutates in place, so if unflattening aliased the original
    ``StateParam`` then calling a layer *inside* a transform would write back into the
    caller's layer — the same class of leak that BUGS.md #6 records for positionally
    passed params.
    """
    layer = pxnn.StatefulLayer(_Counter, 2.0)
    leaves, treedef = jtu.tree_flatten(layer)
    rebuilt = jtu.tree_unflatten(treedef, leaves)

    assert rebuilt.state is not layer.state

    rebuilt(jnp.array(0.0))
    rebuilt(jnp.array(0.0))

    assert layer(jnp.array(0.0)) == 0.0, "advancing the copy must not advance the original"


def test_two_layers_do_not_share_state():
    """Separately constructed layers start from, and keep, separate state.

    ``eqx.nn.StateIndex`` identity is what keys the state store; a wrapper that reused
    one index or one state object across instances would make two BatchNorms in the same
    network accumulate into the same running statistics.
    """
    first = pxnn.StatefulLayer(_Counter, 2.0)
    second = pxnn.StatefulLayer(_Counter, 2.0)

    first(jnp.array(0.0))
    first(jnp.array(0.0))

    assert second(jnp.array(0.0)) == 0.0
    assert first.state is not second.state


# Constructor ##########################################################################


def test_batchnorm_forwards_its_arguments_positionally_to_equinox():
    """``pxnn.BatchNorm``'s signature must map onto ``eqx.nn.BatchNorm``'s, name for name.

    The wrapper forwards seven arguments by position into a signature of eight. Any
    mismatch lands a value in a neighbouring slot — ``eps`` into ``channelwise_affine``,
    ``momentum`` into ``inference`` — and several of those combinations still run.
    """
    layer = pxnn.BatchNorm(3, "batch", EPS, True, MOMENTUM, True, mode="ema")

    assert layer.nn.axis_name.get() == "batch"
    assert layer.nn.eps == EPS
    assert layer.nn.momentum == MOMENTUM
    assert layer.nn.channelwise_affine is True
    assert layer.nn.inference.get() is True


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_batchnorm_defaults_are_trainable_and_in_train_mode():
    """Built with defaults, a BatchNorm has affine parameters and is not in inference mode.

    A layer defaulting to ``inference=True`` would never accumulate statistics, so the
    stored ones would stay at zeros and eval would divide by ``sqrt(eps)``.
    """
    layer = pxnn.BatchNorm(3, "batch")

    assert layer.nn.inference.get() is False
    assert len([p for p in _params(layer) if isinstance(p, pxnn.LayerParam)]) == 2
