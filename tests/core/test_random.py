"""The global random key generator.

`RKG` is the library's one piece of module-level mutable state, seeded from
`time.time_ns()` at import and used as the default argument of every layer and
Vode constructor. Its contract is the one every reproducible experiment rests
on: the same seed must produce the same stream, and drawing must advance the
state so no key is ever handed out twice.

Expectations here come from what a PRNG stream is *supposed* to guarantee and
from `jax.random.split` directly — not from what pcx currently returns.
"""

import jax
import jax.numpy as jnp
import pytest

import pcx


def test_same_seed_gives_the_same_stream():
    """Reproducibility: identical seeds, identical draws, indefinitely."""
    a = pcx.RandomKeyGenerator(seed=0)
    b = pcx.RandomKeyGenerator(seed=0)

    for step in range(5):
        assert jnp.array_equal(a(), b()), f"streams diverged at draw {step}"


def test_different_seeds_give_different_streams():
    a = pcx.RandomKeyGenerator(seed=0)
    b = pcx.RandomKeyGenerator(seed=1)

    assert not jnp.array_equal(a(), b())


def test_consecutive_draws_differ():
    """A generator that returned the same key twice would silently correlate
    every downstream sample, which is the failure this guards against."""
    rkg = pcx.RandomKeyGenerator(seed=0)

    draws = [rkg() for _ in range(8)]

    for i, first in enumerate(draws):
        for j, second in enumerate(draws[i + 1 :], start=i + 1):
            assert not jnp.array_equal(first, second), f"draw {i} == draw {j}"


def test_drawing_advances_the_internal_state():
    rkg = pcx.RandomKeyGenerator(seed=0)

    before = jnp.asarray(rkg.key.get()).copy()
    rkg()
    after = jnp.asarray(rkg.key.get())

    assert not jnp.array_equal(before, after)


def test_seed_resets_the_stream():
    """Re-seeding must rewind, otherwise a test-isolation fixture cannot work."""
    rkg = pcx.RandomKeyGenerator(seed=0)
    first = rkg()
    rkg()
    rkg()

    rkg.seed(0)

    assert jnp.array_equal(rkg(), first)


@pytest.mark.parametrize("n", [2, 3, 8])
def test_batch_draw_returns_n_distinct_keys(n: int):
    """`rkg(n)` is the vmap path: every lane must get its own key, or all lanes
    of a batched model would sample identically."""
    rkg = pcx.RandomKeyGenerator(seed=0)

    keys = rkg(n)

    assert keys.shape[0] == n, f"expected {n} keys, got shape {keys.shape}"
    unique = {tuple(jnp.asarray(k).tolist()) for k in keys}
    assert len(unique) == n, f"only {len(unique)} distinct keys among {n}"


def test_batch_draw_is_uniform_in_n():
    """`rkg(n)` must always yield `n` keys, including `n == 1`.

    Special-casing 1 and returning a bare key would make `keys[0]` a uint32
    rather than a key, and iterating the result would yield the key's two
    integers. Any batched code path hits that the moment a batch size of 1
    occurs — a final partial batch, or a single-example debug run. The bare key
    is reached by calling `rkg()` with no argument instead.
    """
    rkg = pcx.RandomKeyGenerator(seed=0)

    keys = rkg(1)

    assert keys.shape == (1, 2), f"expected shape (1, 2), got {keys.shape}"


def test_batch_draw_also_advances_the_state():
    rkg = pcx.RandomKeyGenerator(seed=0)

    before = jnp.asarray(rkg.key.get()).copy()
    rkg(4)

    assert not jnp.array_equal(before, jnp.asarray(rkg.key.get()))


def test_a_batch_draw_does_not_repeat_a_single_draw():
    """Batched and unbatched draws come off the same stream, so they must not
    collide."""
    rkg = pcx.RandomKeyGenerator(seed=0)

    single = jnp.asarray(rkg()).copy()
    batch = rkg(4)

    for i, k in enumerate(batch):
        assert not jnp.array_equal(single, k), f"batch key {i} repeats an earlier draw"


def test_keys_are_usable_by_jax_random():
    """The generator is only useful if what it yields is a valid jax key."""
    rkg = pcx.RandomKeyGenerator(seed=0)

    sample = jax.random.normal(rkg(), (16,))

    assert sample.shape == (16,)
    assert jnp.all(jnp.isfinite(sample))


def test_independent_generators_do_not_share_state():
    """Constructing a second generator must not disturb the first — otherwise a
    test using a private generator would still perturb the global one."""
    a = pcx.RandomKeyGenerator(seed=0)
    expected = [jnp.asarray(a()).copy() for _ in range(3)]

    a.seed(0)
    b = pcx.RandomKeyGenerator(seed=99)
    got = []
    for _ in range(3):
        got.append(jnp.asarray(a()).copy())
        b()

    for i, (want, have) in enumerate(zip(expected, got, strict=True)):
        assert jnp.array_equal(want, have), f"draw {i} was perturbed by the other generator"


def test_state_is_a_pytree_leaf():
    """RKG rides through transforms as a traced kwarg, so its state has to be a
    dynamic pytree leaf rather than static metadata."""
    rkg = pcx.RandomKeyGenerator(seed=0)

    leaves = jax.tree_util.tree_leaves(rkg)

    assert len(leaves) == 1, f"expected exactly one dynamic leaf, got {len(leaves)}"


def test_global_rkg_is_seeded_and_reproducible_across_the_fixture():
    """The autouse fixture in conftest reseeds the global generator, so this
    test and the next must see the same first draw."""
    assert jnp.array_equal(pcx.RKG(), _first_global_draw())


def _first_global_draw():
    pcx.RKG.seed(0)
    return jnp.asarray(pcx.RKG()).copy()


def test_global_rkg_isolation_fixture_actually_isolates():
    """Deliberately burn the global stream. The autouse fixture must undo it, so
    the assertion above still holds on the next test."""
    for _ in range(10):
        pcx.RKG()

    pcx.RKG.seed(0)
    assert jnp.array_equal(pcx.RKG(), _first_global_draw())
