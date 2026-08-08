"""Shared pytest configuration for the pcx test suite.

Tests must be reproducible and must not depend on an accelerator being present,
so the CPU backend is pinned before JAX is imported anywhere.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import pytest


@pytest.fixture
def key() -> jax.Array:
    """A fixed PRNG key, so every test that draws randomness is reproducible."""
    return jax.random.PRNGKey(0)
