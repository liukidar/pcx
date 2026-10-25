"""The `pxu.step` context manager.

Every training loop in the library is a stack of `pxu.step` blocks, so its
contract is load-bearing: set a status for the duration of the block, clear the
requested caches, and leave the model in a clean state afterwards. A block that
leaks its status silently changes which ruleset the *next* computation runs
under, which shows up as wrong energies rather than as an error.
"""

import jax.numpy as jnp
import pytest

import pcx
import pcx.functional as pxf
import pcx.predictive_coding as pxc
import pcx.utils as pxu


class Model(pxc.EnergyModule):
    """A module with a Vode, so status and cache clearing both have an effect."""

    def __init__(self):
        super().__init__()
        self.vode = pxc.Vode()

    def __call__(self, u):
        return self.vode(u)


@pytest.fixture
def model():
    m = Model()
    m(jnp.array([1.0, 2.0]))
    return m


def statuses(module):
    """The status of the module and every EnergyModule beneath it."""
    return {module.status} | {sub.status for sub in module.submodules(cls=pxc.EnergyModule)}


def test_status_is_set_for_the_duration_of_the_block(model):
    """The block exists to make a status active while its body runs."""
    with pxu.step(model, pxc.STATUS.INIT):
        assert model.status == pxc.STATUS.INIT


def test_status_propagates_to_submodules(model):
    """A Vode reads its own status, not its parent's, so the status has to reach
    every EnergyModule or the ruleset silently fails to apply."""
    with pxu.step(model, pxc.STATUS.INIT):
        assert statuses(model) == {pxc.STATUS.INIT}


@pytest.mark.parametrize("status", [pxc.STATUS.INIT, "custom", None])
def test_the_requested_status_is_the_one_applied(model, status):
    with pxu.step(model, status):
        assert model.status == status


def test_an_explicit_exit_status_is_applied(model):
    """The two-tuple form names the exit status explicitly, and the tutorials rely
    on it."""
    with pxu.step(model, (pxc.STATUS.INIT, None)):
        assert model.status == pxc.STATUS.INIT

    assert model.status is None


def test_clear_params_runs_after_the_block(model):
    """A scalar `clear_params` is documented as clearing *after* the body, so the
    body still sees the cache it populated."""
    model.vode.cache["probe"] = jnp.array([1.0])

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        # Assert the value survives, not merely that the container is non-None:
        # a cache that was cleared and rebuilt empty would pass the weaker check.
        assert jnp.array_equal(model.vode.cache["probe"], jnp.array([1.0]))

    assert pcx.get(model.vode.cache) is None


def test_clear_params_tuple_clears_before_and_after(model):
    """The tuple form is `(before, after)` — easy to misread as "clear both of
    these", so pin which slot means what."""
    model.vode.cache["probe"] = jnp.array([1.0])

    with pxu.step(model, clear_params=(pxc.VodeParam.Cache, None)):
        assert pcx.get(model.vode.cache) is None, "the 'before' slot did not clear on entry"


def test_status_is_restored_after_the_block(model):
    """Leaving the block must leave the status as it was found.

    Otherwise the canonical single-status form documented in the tutorials
    leaves the model in `'init'` forever, and any energy computed between that
    block and the next runs under the initialisation ruleset, which copies `u`
    into `h` and drives the energy to zero.
    """
    assert model.status is None

    with pxu.step(model, pxc.STATUS.INIT):
        pass

    assert model.status is None, f"status leaked out of the block as {model.status!r}"


def test_an_exception_in_the_body_still_restores_the_model():
    """A failing body must not leave the model corrupted.

    Without `try/finally` an exception would skip the cache clear and the status
    reset, so the model would keep the block's status. Under pytest the next test
    inherits it, which turns one real failure into a cascade of unrelated ones.
    """
    model = Model()
    model(jnp.array([1.0, 2.0]))

    with pytest.raises(RuntimeError):
        with pxu.step(model, (pxc.STATUS.INIT, None), clear_params=pxc.VodeParam.Cache):
            raise RuntimeError("deliberate")

    assert model.status is None, f"status left as {model.status!r} after the body raised"


def test_nested_blocks_restore_the_outer_status(model):
    """Nesting is natural — an inference loop inside a training block — so the
    inner block must not clobber the status the outer one established."""
    with pxu.step(model, pxc.STATUS.INIT):
        with pxu.step(model, "inner"):
            assert model.status == "inner"

        assert model.status == pxc.STATUS.INIT, f"inner block left the status as {model.status!r}"

    assert model.status is None


def test_an_explicit_exit_status_wins_over_restoring(model):
    """The two-tuple form is a request for a specific exit status, so it is
    applied even when the status on entry was different."""
    with pxu.step(model, pxc.STATUS.INIT):
        with pxu.step(model, ("inner", None)):
            pass

        assert model.status is None


def test_a_module_reachable_twice_is_restored_to_its_entry_status():
    """`tree_apply` visits a shared module once per reference, so the second visit
    sees the block's status, not the original one. Restoring must still land on the
    status the module had on entry."""
    vode = pxc.Vode()

    class Shared(pxc.EnergyModule):
        def __init__(self):
            super().__init__()
            self.a = vode
            self.b = vode

    m = Shared()
    vode.status = "outer"

    with pxu.step(m, pxc.STATUS.INIT):
        assert vode.status == pxc.STATUS.INIT

    assert vode.status == "outer"


def test_a_block_inside_a_scan_body_keeps_the_carry_structure(model):
    """The status is static pytree metadata, so `jax.lax.scan` requires it to be
    the same on entry to and exit from the body. A block that does not restore the
    status changes the carry's structure and the scan is rejected."""
    model.status = "outer"

    def body(_, *, model):
        with pxu.step(model, pxc.STATUS.INIT):
            assert model.status == pxc.STATUS.INIT
        return (), None

    pxf.scan(body, length=3)(model=model)

    assert model.status == "outer"
