"""Checkpoint save/load.

The contract is simple and absolute: what you load must be what you saved. A
checkpoint that silently loads the wrong weights costs a training run and is
invisible until results look odd, so the tests here assert exact equality rather
than closeness, and check the failure modes that would corrupt a restore.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import pcx
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
import pcx.utils as pxu


class Net(pcx.Module):
    """Two layers, so ordering and key derivation are actually exercised."""

    def __init__(self, rkg, in_dim=3, hidden=4, out_dim=2):
        self.a = pxnn.Linear(in_dim, hidden, rkg=rkg)
        self.b = pxnn.Linear(hidden, out_dim, rkg=rkg)


@pytest.fixture
def net(rkg):
    return Net(rkg)


@pytest.fixture
def other_net():
    """A second, independently initialised net — the load target."""
    return Net(pcx.RandomKeyGenerator(seed=12345))


@pytest.fixture
def ckpt(tmp_path):
    return str(tmp_path / "model")


def weights(model):
    """Every LayerParam value, keyed by path, as concrete arrays."""
    import jax.tree_util as jtu

    return {
        jtu.keystr(k): np.asarray(pcx.get(p))
        for k, p in jtu.tree_flatten_with_path(model, is_leaf=lambda x: isinstance(x, pxnn.LayerParam))[0]
        if isinstance(p, pxnn.LayerParam)
    }


def assert_weights_identical(actual, expected):
    """Bit-exact, not approximate: a checkpoint that round-trips lossily is broken."""
    assert actual.keys() == expected.keys(), f"parameter sets differ: {actual.keys()} vs {expected.keys()}"
    for path, want in expected.items():
        np.testing.assert_array_equal(actual[path], want, err_msg=f"{path} changed across the round trip")


def weights_differ(a, b) -> bool:
    """True if any weight differs. Dicts of arrays cannot be compared with `!=`."""
    return a.keys() != b.keys() or any(not np.array_equal(a[k], b[k]) for k in a)


def test_round_trip_restores_every_weight_exactly(net, other_net, ckpt):
    """The whole point of the module. Exact equality, since float32 arrays are
    written and read verbatim — any drift means a conversion bug."""
    saved = weights(net)
    assert weights_differ(weights(other_net), saved), "fixture error: the two nets must start different"

    pxu.save_params(net, ckpt)
    pxu.load_params(other_net, ckpt)

    assert_weights_identical(weights(other_net), saved)


def test_round_trip_survives_an_explicit_npz_extension(net, other_net, ckpt):
    """`save_params` appends `.npz` via numpy while `load_params` appends it
    explicitly, so the two can disagree about what the path means. Both
    spellings must name the same file."""
    pxu.save_params(net, ckpt + ".npz")
    pxu.load_params(other_net, ckpt + ".npz")

    assert_weights_identical(weights(other_net), weights(net))


def test_loading_does_not_disturb_parameters_outside_the_filter(rkg, ckpt):
    """The default filter is LayerParam, so Vode states must be left alone —
    otherwise a restore would silently reset inference state."""

    class WithVode(pcx.Module):
        def __init__(self):
            self.layer = pxnn.Linear(2, 2, rkg=rkg)
            self.vode = pxc.Vode()

    source, target = WithVode(), WithVode()
    target.vode.h.set(jnp.array([7.0, 7.0]))

    pxu.save_params(source, ckpt)
    pxu.load_params(target, ckpt)

    np.testing.assert_array_equal(np.asarray(pcx.get(target.vode.h)), np.array([7.0, 7.0]))


def test_a_non_default_filter_saves_the_selected_parameters(rkg, ckpt):
    """`filter=VodeParam` is the documented way to checkpoint inference state."""

    class WithVode(pcx.Module):
        def __init__(self):
            self.layer = pxnn.Linear(2, 2, rkg=rkg)
            self.vode = pxc.Vode()

    source, target = WithVode(), WithVode()
    source.vode.h.set(jnp.array([3.0, -4.0]))
    target.vode.h.set(jnp.array([0.0, 0.0]))

    pxu.save_params(source, ckpt, filter=pxc.VodeParam)
    pxu.load_params(target, ckpt, filter=pxc.VodeParam)

    np.testing.assert_array_equal(np.asarray(pcx.get(target.vode.h)), np.array([3.0, -4.0]))


def test_missing_parameter_raises_key_error(net, other_net, ckpt):
    """A checkpoint that does not cover the model must fail loudly. Silently
    leaving a layer at its random initialisation is the worst outcome."""
    pxu.save_params(net.a, ckpt)  # only the first layer

    with pytest.raises(KeyError):
        pxu.load_params(other_net, ckpt)


def test_keys_are_stable_attribute_paths(net, ckpt):
    """Checkpoint keys are `jtu.keystr` attribute paths, so renaming or
    reordering an attribute invalidates every existing checkpoint. Pinning the
    exact strings makes that breakage visible in a diff instead of at restore
    time, months later."""
    pxu.save_params(net, ckpt)

    with np.load(ckpt + ".npz") as f:
        assert set(f.files) == {".a.nn.weight", ".a.nn.bias", ".b.nn.weight", ".b.nn.bias"}


@pytest.mark.bug("BUGS.md#26: load_params does no shape check, so a mismatched checkpoint overwrites silently")
def test_shape_mismatch_is_rejected(rkg, ckpt):
    """Loading a checkpoint from a differently-shaped model must not succeed.
    It currently overwrites the parameter with the wrong shape, which surfaces
    much later as an inscrutable broadcasting error deep in a training step."""
    source = pxnn.Linear(5, 7, rkg=rkg)
    target = pxnn.Linear(2, 3, rkg=pcx.RandomKeyGenerator(seed=1))
    original = np.asarray(pcx.get(target.nn.weight))

    pxu.save_params(source, ckpt)
    try:
        pxu.load_params(target, ckpt)
    except (ValueError, TypeError, KeyError):
        return  # rejected, which is the correct outcome

    loaded = np.asarray(pcx.get(target.nn.weight))
    assert loaded.shape == original.shape, f"silently loaded a {loaded.shape} weight into a {original.shape} parameter"


@pytest.mark.bug("BUGS.md#10: duplicate refs are written as None, so np.load refuses the object array")
def test_round_trip_preserves_shared_parameters(rkg, ckpt):
    """A model using `pxnn.shared` must checkpoint and restore like any other.

    Duplicate references are written as a literal `None`, which numpy stores as
    a `dtype=object` array; `load_params` then calls `np.load` with the default
    `allow_pickle=False` and refuses it. Every checkpoint of a weight-tied model
    is therefore unrecoverable.
    """
    layer = pxnn.Linear(2, 2, rkg=rkg)
    source = [layer, pxnn.shared(layer)]
    target_layer = pxnn.Linear(2, 2, rkg=pcx.RandomKeyGenerator(seed=1))
    target = [target_layer, pxnn.shared(target_layer)]

    pxu.save_params(source, ckpt)
    pxu.load_params(target, ckpt)

    np.testing.assert_array_equal(np.asarray(pcx.get(target[0].nn.weight)), np.asarray(pcx.get(layer.nn.weight)))
    assert target[0].nn.weight is target[1].nn.weight, "sharing was not preserved across the round trip"
