"""Smoke tests proving the package imports and its core primitives round-trip.

These exist so the CI harness itself is verified on every supported platform.
The substantive test suite is built out separately.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import pcx


def test_public_api_is_exported():
    for name in ("Param", "BaseParam", "Module", "BaseModule", "get", "set", "static"):
        assert hasattr(pcx, name), f"pcx.{name} is missing from the public API"


def test_param_holds_and_updates_a_value():
    param = pcx.Param(jnp.zeros(3))

    assert jnp.array_equal(pcx.get(param), jnp.zeros(3))

    param.set(jnp.ones(3))
    assert jnp.array_equal(pcx.get(param), jnp.ones(3))


def test_module_round_trips_through_the_pytree_registry():
    class Model(pcx.Module):
        def __init__(self):
            self.w = pcx.Param(jnp.ones(2))

    model = Model()
    leaves, treedef = jtu.tree_flatten(model)
    restored = jtu.tree_unflatten(treedef, leaves)

    assert isinstance(restored, Model)
    assert jnp.array_equal(pcx.get(restored.w), jnp.ones(2))


def test_random_key_generator_is_deterministic(key: jax.Array):
    del key  # this test seeds pcx's own generator rather than using the fixture

    # __call__ returns a tuple of keys when asked for several, so normalise before comparing.
    first = jnp.asarray(pcx.RandomKeyGenerator(seed=0)())
    second = jnp.asarray(pcx.RandomKeyGenerator(seed=0)())
    other = jnp.asarray(pcx.RandomKeyGenerator(seed=1)())

    assert jnp.array_equal(first, second)
    assert not jnp.array_equal(first, other)
