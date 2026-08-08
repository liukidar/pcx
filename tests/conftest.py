"""Shared pytest configuration for the pcx test suite.

Tests must be reproducible and must not depend on an accelerator being present,
so the CPU backend is pinned before JAX is imported anywhere.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import pytest

import pcx

#: Seed every test starts from. Fixed so failures reproduce exactly.
SEED = 0


@pytest.fixture(autouse=True)
def _isolate_global_rkg():
    """Reseed pcx's global RNG before each test and restore its key afterwards.

    ``pcx.RKG`` is module-level state seeded from ``time.time_ns()`` at import
    (``pcx/core/_random.py``), and it is the default argument of every layer and
    Vode constructor. Any test that builds a model therefore advances a stream
    shared with every other test, which makes the suite order-dependent and
    non-reproducible. Reseeding here removes that coupling.

    Restoring the key on the way out also contains a known failure mode: the
    transforms swap ``RKG.key`` for a traced value without ``try/finally``, so a
    test whose transformed function raises can leave a tracer installed globally
    and poison every subsequent test in the process.
    """
    pcx.RKG.seed(SEED)
    yield
    pcx.RKG.seed(SEED)


@pytest.fixture
def key() -> jax.Array:
    """A fixed PRNG key, so every test that draws randomness is reproducible."""
    return jax.random.PRNGKey(SEED)


@pytest.fixture
def rkg() -> pcx.RandomKeyGenerator:
    """A private key generator, for tests that must not touch the global one."""
    return pcx.RandomKeyGenerator(seed=SEED)


def assert_allclose(actual, desired, *, rtol=1e-6, atol=1e-6, err_msg=""):
    """float32-appropriate closeness check with a readable failure message."""
    import numpy as np

    np.testing.assert_allclose(np.asarray(actual), np.asarray(desired), rtol=rtol, atol=atol, err_msg=err_msg)


def tree_allclose(actual, desired, *, rtol=1e-6, atol=1e-6):
    """Compare two pytrees leaf-by-leaf, requiring identical structure."""
    a_leaves, a_def = jax.tree_util.tree_flatten(actual)
    d_leaves, d_def = jax.tree_util.tree_flatten(desired)
    assert a_def == d_def, f"pytree structures differ:\n{a_def}\n!=\n{d_def}"
    for i, (a, d) in enumerate(zip(a_leaves, d_leaves, strict=True)):
        assert_allclose(a, d, rtol=rtol, atol=atol, err_msg=f"leaf {i} differs")


def count_leaves(tree) -> int:
    """Number of dynamic leaves, the quantity jax transformations care about."""
    return len(jax.tree_util.tree_leaves(tree))


__all__ = ["SEED", "assert_allclose", "count_leaves", "jnp", "tree_allclose"]
