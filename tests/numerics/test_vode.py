"""Behaviour of `pcx.predictive_coding.Vode` and `EnergyModule`.

A Vode is a value node: it holds a state ``h``, receives a prediction ``u``, and
contributes an energy term. Its behaviour is driven by a status-dependent ruleset,
and its energy is cached. All three of those — the state machine, the cache and the
energy accumulation — are places where a defect produces a plausible-looking number
rather than an error, so the expectations here are written out from the closed form
``0.5 * (h - u)**2`` and from the documented contract of each method.
"""

import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import pytest
from conftest import assert_allclose

import pcx
import pcx.predictive_coding as pxc
import pcx.utils as pxu

H = jnp.array([[1.0, 2.0], [3.0, 4.0], [0.0, -1.0]])
U = jnp.zeros_like(H)

#: Per-sample squared-error energy of (H, U), written out by hand: 0.5 * sum_j (h - u)**2.
PER_SAMPLE_ENERGY = jnp.array([0.5 * (1.0 + 4.0), 0.5 * (9.0 + 16.0), 0.5 * (0.0 + 1.0)])


class _Net(pxc.EnergyModule):
    """An energy module holding a list of Vodes, as every pcx network does."""

    def __init__(self, n: int):
        super().__init__()
        self.vodes = [pxc.Vode() for _ in range(n)]


class _NoVodes(pxc.EnergyModule):
    """An energy module with no value nodes at all — a pure feed-forward stack."""

    def __init__(self):
        super().__init__()
        self.scale = pcx.Param(jnp.ones(()))


# Energy accumulation ##################################################################


@pytest.mark.bug("EnergyModule.energy() reduces over an empty iterable and raises TypeError")
def test_energy_of_a_module_with_no_vodes_is_zero():
    """The total energy of a module is the sum over its Vodes, and the sum over an
    empty collection is zero.

    This is not a corner case: a purely feed-forward branch, or a model whose Vodes
    all live one level deeper, has no direct Vode children. Raising here forces every
    caller to special-case a quantity that has a perfectly good value.
    """
    assert_allclose(_NoVodes().energy(), 0.0)


def test_module_energy_is_the_sum_of_its_vode_energies():
    """Additivity: ``E_total = sum_i E_i``. The variational free energy of a predictive
    coding network is defined as this sum; if the accumulation drops or double-counts
    a node, the reported energy no longer matches the objective being minimised.
    """
    net = _Net(2)
    net.vodes[0].h.set(H)
    net.vodes[0](U)
    net.vodes[1].h.set(2.0 * H)
    net.vodes[1](U)

    expected = (0.5 * (H - U) ** 2).sum() + (0.5 * (2.0 * H - U) ** 2).sum()

    assert_allclose(net.energy(), expected)


def test_vode_energy_equals_the_closed_form_total():
    """A single node's energy is ``0.5 * sum (h - u)**2``, computed here by hand."""
    vode = pxc.Vode()
    vode.h.set(H)
    vode(U)

    assert_allclose(jnp.sum(vode.energy()), jnp.sum(0.5 * (H - U) ** 2))


@pytest.mark.bug("Vode.energy() collapses to a scalar once __call__ has recorded the input shape")
def test_energy_is_the_same_whether_u_was_set_by_call_or_by_set():
    """``Vode.__call__`` is documented as equivalent to ``vode.set("u", u).get("h")``,
    so the energy must not depend on which of the two was used.

    It does: ``__call__`` records the input shape, and `Vode.energy` uses that record
    to decide between returning the documented per-sample vector of shape
    ``(batch_size,)`` and summing everything into a scalar. The total is the same, but
    any analysis that reads per-sample energies gets a scalar from one call path and a
    vector from the other.
    """
    via_set = pxc.Vode()
    via_set.h.set(H)
    via_set.set("u", U)

    via_call = pxc.Vode()
    via_call.h.set(H)
    via_call(U)

    assert_allclose(via_set.energy(), PER_SAMPLE_ENERGY)
    assert via_call.energy().shape == PER_SAMPLE_ENERGY.shape, (
        f"expected per-sample energies of shape {PER_SAMPLE_ENERGY.shape}, got shape {via_call.energy().shape}"
    )
    assert_allclose(via_call.energy(), PER_SAMPLE_ENERGY)


# Cache ################################################################################


@pytest.mark.bug(
    "clear_params sets the cache dict to None, so every subsequent read raises instead of seeing an empty cache"
)
def test_a_cleared_cache_reads_as_empty_rather_than_raising():
    """Clearing a cache must leave it empty and readable, not broken.

    ``pxu.step(..., clear_params=VodeParam.Cache)`` runs around every inference step, so
    a cleared cache is the normal state of a Vode between steps. `ParamDict.__setitem__`
    already treats a cleared dict as empty and recreates it on write; the read path
    (``__contains__`` and ``get``) does not. `Vode.energy` is the immediate casualty —
    its first statement is ``if "E" not in self.cache`` — so recomputing the energy
    after a clear raises ``TypeError`` instead of recomputing.
    """
    vode = pxc.Vode()
    vode.h.set(H)
    vode.set("u", U)
    vode.energy()

    vode.clear_params(pxc.VodeParam.Cache)

    assert "E" not in vode.cache, "a cleared cache still reports a cached energy"
    assert vode.get("u") is None, "a cleared cache still reports a cached activation"


def test_energy_recomputes_after_the_cache_is_cleared_and_u_is_set_again():
    """The supported flow — clear the cache, run the forward pass again, read the
    energy — must produce the energy of the *new* prediction, not a stale cached one.

    A cache that survived clearing would freeze the energy at its first value, so
    inference would appear to make no progress at all.
    """
    vode = pxc.Vode()
    vode.h.set(H)
    vode.set("u", U)
    assert_allclose(vode.energy(), PER_SAMPLE_ENERGY)

    vode.clear_params(pxc.VodeParam.Cache)
    new_u = H - 1.0
    vode.set("u", new_u)

    assert_allclose(vode.energy(), jnp.full((3,), 0.5 * 2.0))


# Ruleset state machine ################################################################


def test_init_status_forward_initialises_h_from_u():
    """Default ruleset: under ``STATUS.INIT`` the rule ``h, u <- u`` fires, so setting the
    incoming activation also seeds the node's value with it.

    This is forward initialisation — the network's first guess for every node is its
    feed-forward prediction. Without it, inference starts from whatever ``h`` happened
    to hold and the first energy of every batch is meaningless.
    """
    vode = pxc.Vode()
    vode.status = pxc.STATUS.INIT

    vode(U + 0.5)

    assert jnp.array_equal(vode.h.get(), U + 0.5), "h was not forward-initialised from u"
    assert jnp.array_equal(vode.get("u"), U + 0.5)


def test_default_status_leaves_h_untouched_when_u_is_set():
    """Outside the init phase the rule must NOT fire: ``h`` is the free variable that
    inference optimises, so overwriting it with ``u`` would collapse the prediction
    error to zero and stop learning entirely.
    """
    vode = pxc.Vode()
    vode.h.set(H)

    vode(U + 0.5)

    assert jnp.array_equal(vode.h.get(), H), "h was overwritten outside the init status"
    assert jnp.array_equal(vode.get("u"), U + 0.5)


@pytest.mark.parametrize("status", ["train", "eval", "inference"])
def test_a_non_matching_status_does_not_fire_the_init_rule(status: str):
    """The ruleset keys are regular expressions matched against the status. Only a
    status matching ``"init"`` may forward-initialise; any other phase must leave ``h``
    alone.
    """
    vode = pxc.Vode()
    vode.h.set(H)
    vode.status = status

    vode(U + 0.5)

    assert jnp.array_equal(vode.h.get(), H), f"status {status!r} fired the init rule"


def test_init_status_energy_is_zero_because_h_equals_u():
    """Immediate consequence of forward initialisation: right after the init pass the
    prediction error is exactly zero at every node, so the network starts inference at
    the bottom of its own energy landscape.
    """
    vode = pxc.Vode()
    vode.status = pxc.STATUS.INIT
    vode(H)

    assert bool(jnp.all(vode.energy() == 0.0))


# Frozen state #########################################################################


def test_a_frozen_h_is_bit_identical_after_an_inference_step():
    """A frozen value node is pinned: during supervised inference the output node holds
    the label, and it must not move.

    ``pxu.M_hasnot(VodeParam, frozen=True)`` is the mask that excludes it. If a frozen
    node drifts even slightly, the network is no longer being clamped to its target and
    the whole supervised setup degenerates.
    """
    net = _Net(2)
    for vode in net.vodes:
        vode.h.set(H)
        vode(U)
    net.vodes[1].h.frozen = True

    pinned = net.vodes[1].h.get()

    masked = pxu.M_hasnot(pxc.VodeParam, frozen=True)(net)
    optim = pxu.Optim(lambda: optax.sgd(0.5), masked)
    grads = jtu.tree_map(
        lambda p: type(p)(jnp.ones_like(p.get())),
        masked,
        is_leaf=lambda x: isinstance(x, pcx.BaseParam),
    )
    optim.step(masked, grads)

    assert jnp.array_equal(net.vodes[1].h.get(), pinned), "a frozen h moved during inference"
    assert_allclose(net.vodes[0].h.get(), H - 0.5)
