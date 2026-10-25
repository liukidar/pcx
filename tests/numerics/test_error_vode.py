"""`ErrorVode` and `apply_h_updates`.

An ErrorVode stores the prediction error ``e = h - u`` and reconstructs
``h = sg(u) + e``. The contract is that this is a change of what is *stored*, not
of the model: energies, the local PC gradients and (with `apply_h_updates`) the
inference trajectories are those of the same network built from Vodes. The
oracles are therefore the Vode network itself and, for the precision claims, the
local PC gradients written out in float64 NumPy:

    dF/dh_l = e_l - W_{l+1}^T (f'(a_{l+1}) * e_{l+1}),   a_{l+1} = W_{l+1} h_l + b_{l+1}.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from conftest import assert_allclose

import pcx
import pcx.functional as pxf
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
import pcx.utils as pxu

DIMS = (3, 5, 4, 2)
B = 6


class MLP(pxc.EnergyModule):
    """x -> [Linear -> act -> node] x 2 -> Linear -> clamped output Vode."""

    def __init__(self, node: type[pxc.Vode], act: Callable[[jax.Array], jax.Array], seed: int = 0):
        super().__init__()
        rkg = pcx.RandomKeyGenerator(seed)
        self.act = pcx.static(act)
        self.layers = [pxnn.Linear(DIMS[i], DIMS[i + 1], rkg=rkg) for i in range(len(DIMS) - 1)]
        # Vodes or ErrorVodes: the tests read `.h` or `.e` depending on the variant.
        self.nodes: list[Any] = [node() for _ in range(len(DIMS) - 2)]
        self.out = pxc.Vode()
        self.out.h.frozen = True

    def __call__(self, x, y):
        for layer, node in zip(self.layers[:-1], self.nodes, strict=True):
            x = node(self.act(layer(x)))
        self.out(self.layers[-1](x))
        if y is not None:
            self.out.set("h", y)
        return self.out.get("u")


VMASK = pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0))


@pxf.vmap(VMASK, in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model):
    return model(x, y)


@pxf.vmap(VMASK, in_axes=(0,), out_axes=(None, None), axis_name="batch")
def energy(x, *, model):
    model(x, None)
    per_node = jnp.stack([n.energy() for n in [*model.nodes, model.out]])
    per_node = jax.lax.psum(per_node, "batch")
    return per_node.sum(), per_node


def _state_mask():
    return pxu.M_hasnot(pxc.VodeParam, frozen=True)


def init(model, x, y):
    with pxu.step(model, (pxc.STATUS.INIT, None), clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)


def state_grads(model, x):
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (_, (per_node,)), g = pxf.value_and_grad(_state_mask().to((False, True)), has_aux=True)(energy)(x, model=model)
    return per_node, g["model"]


def weight_grads(model, x):
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to((False, True)), has_aux=True)(energy)(x, model=model)
    return [jnp.asarray(pcx.get(layer.nn.weight)) for layer in g["model"].layers]


def values(model, x):
    """h of every hidden node, recomputed by a forward pass from the current states."""
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        forward(x, None, model=model)
        return [jnp.asarray(n.get("h")) for n in model.nodes]


def data(scale=1.0):
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    return scale * jax.random.normal(k1, (B, DIMS[0])), jax.random.normal(k2, (B, DIMS[-1]))


def pair(act=jnp.tanh, errors_scale=0.3):
    """A Vode and an ErrorVode network with the same weights, at the same state h = u + eps."""
    x, y = data()
    plain, err = MLP(pxc.Vode, act), MLP(pxc.ErrorVode, act)
    init(plain, x, y)
    init(err, x, y)

    keys = jax.random.split(jax.random.PRNGKey(2), len(plain.nodes))
    for k, pn, en in zip(keys, plain.nodes, err.nodes, strict=True):
        eps = errors_scale * jax.random.normal(k, pn.h.get().shape)
        en.e.set(eps)
    # The ErrorVode's values, recomputed from its errors, are the Vode network's states.
    for pn, h in zip(plain.nodes, values(err, x), strict=True):
        pn.h.set(h)

    return x, y, plain, err


# Structure ############################################################################


def test_forward_initialisation_sets_a_zero_error():
    x, y = data()
    model = MLP(pxc.ErrorVode, jnp.tanh)

    with pxu.step(model, pxc.STATUS.INIT):
        forward(x, y, model=model)
        for node in model.nodes:
            assert_allclose(node.e.get(), jnp.zeros_like(node.cache["u"]))
            assert_allclose(node.get("h"), node.get("u"))


def test_an_error_vode_cannot_be_clamped():
    """A clamped value is fixed while u moves, so it is not a fixed error."""
    with pytest.raises(ValueError, match="ErrorVode"):
        pxc.ErrorVode().set("h", jnp.ones(2))


def test_the_energy_is_half_the_squared_stored_error():
    """The residual is e + (sg(u) - u), whose value is e exactly."""
    node = pxc.ErrorVode()
    node.e.set(jnp.array([[1e-3, -2e-3]]))
    node(jnp.array([[1e3, 1e3]]))

    np.testing.assert_array_equal(np.asarray(node.residual()), np.array([[1e-3, -2e-3]], dtype=np.float32))


# Equivalence with a Vode network #######################################################


def test_energies_match_a_vode_network_at_the_same_state():
    x, _, plain, err = pair()

    e_plain, _ = state_grads(plain, x)
    e_err, _ = state_grads(err, x)

    assert_allclose(e_err, e_plain, rtol=1e-5, atol=1e-6)


def test_the_error_gradient_is_the_local_pc_state_gradient():
    """dF/de_l of the ErrorVode network equals dF/dh_l of the Vode network: the
    stop-gradient keeps it local instead of flowing through the layers above."""
    x, _, plain, err = pair()

    _, g_plain = state_grads(plain, x)
    _, g_err = state_grads(err, x)

    for gp, ge in zip(g_plain.nodes, g_err.nodes, strict=True):
        assert_allclose(ge.e.get(), gp.h.get(), rtol=1e-5, atol=1e-6)


def test_weight_gradients_match_a_vode_network():
    x, _, plain, err = pair()

    for wp, we in zip(weight_grads(plain, x), weight_grads(err, x), strict=True):
        assert_allclose(we, wp, rtol=1e-5, atol=1e-6)


# apply_h_updates ######################################################################


def test_apply_h_updates_on_vodes_is_a_plain_update():
    x, _, plain, _ = pair()
    _, g = state_grads(plain, x)
    before = [n.h.get() for n in plain.nodes]
    updates = jax.tree.map(lambda a: -0.1 * a, g)

    with pxu.step(plain, clear_params=pxc.VodeParam.Cache):
        pxc.apply_h_updates(forward, updates, x, None, model=plain)

    for n, h0, gn in zip(plain.nodes, before, g.nodes, strict=True):
        assert_allclose(n.h.get(), h0 - 0.1 * gn.h.get())


def _step_error(order, s):
    """max |h_after - (h + s d)| over nodes after one apply_h_updates of size s."""
    x, _, _, err = pair()
    h0 = values(err, x)
    _, g = state_grads(err, x)
    updates = jax.tree.map(lambda a: -s * a, g)

    with pxu.step(err, clear_params=pxc.VodeParam.Cache):
        pxc.apply_h_updates(forward, updates, x, None, model=err, order=order)

    h1 = values(err, x)
    return max(float(jnp.max(jnp.abs(a - (b - s * gn.e.get())))) for a, b, gn in zip(h1, h0, g.nodes, strict=True))


@pytest.mark.parametrize("order", [1, 2])
def test_apply_h_updates_moves_h_by_the_update_up_to_the_taylor_remainder(order):
    """The h-space step is exact up to O(s^(order+1)): halving the step divides the
    error by about 2^(order+1)."""
    big, small = _step_error(order, 2.0), _step_error(order, 1.0)
    ratio = big / small

    assert 0.7 * 2 ** (order + 1) < ratio < 1.3 * 2 ** (order + 1), f"error ratio {ratio} for order {order}"


def test_apply_h_updates_is_exact_for_a_linear_network():
    """With linear layers the Taylor expansion is exact at any order."""
    x, _, _, err = pair(act=lambda a: a)
    h0 = values(err, x)
    _, g = state_grads(err, x)
    updates = jax.tree.map(lambda a: -0.5 * a, g)

    with pxu.step(err, clear_params=pxc.VodeParam.Cache):
        pxc.apply_h_updates(forward, updates, x, None, model=err, order=1)

    for a, b, gn in zip(values(err, x), h0, g.nodes, strict=True):
        assert_allclose(a, b - 0.5 * gn.e.get(), rtol=1e-5, atol=1e-6)


def _run(model, x, T, h_steps: bool):
    @pxf.jit()
    def infer(x, *, model, optim_h):
        traj = []
        for _ in range(T):
            per_node, g = state_grads(model, x)
            traj.append(per_node.sum())
            if h_steps:
                updates = optim_h.step(model, g, apply_updates=False)
                with pxu.step(model, clear_params=pxc.VodeParam.Cache):
                    pxc.apply_h_updates(forward, updates, x, None, model=model)
            else:
                optim_h.step(model, g)
        return jnp.stack(traj)

    optim_h = pxu.Optim(lambda: optax.sgd(0.05, momentum=0.5))
    optim_h.init(_state_mask()(model))
    return infer(x, model=model, optim_h=optim_h)


def test_inference_trajectory_matches_a_vode_network():
    """SGD with momentum under jit: the ErrorVode network with h-space steps follows
    the Vode network's energy trajectory, and ends at the same states."""
    x, _, plain, err = pair()

    traj_plain = _run(plain, x, 20, h_steps=False)
    traj_err = _run(err, x, 20, h_steps=True)

    assert_allclose(traj_err, traj_plain, rtol=1e-4, atol=1e-6)
    for a, pn in zip(values(err, x), plain.nodes, strict=True):
        assert_allclose(a, pn.h.get(), rtol=1e-4, atol=1e-5)


def test_stepping_the_errors_directly_is_a_different_dynamics():
    """`optim.step` on ErrorVodes moves h_l by the update plus the change of u_l
    caused by the layers below. Pinned so the difference stays documented."""
    x, _, plain, err = pair()

    traj_plain = _run(plain, x, 20, h_steps=False)
    traj_err = _run(err, x, 20, h_steps=False)

    assert float(jnp.max(jnp.abs(traj_err - traj_plain))) > 1e-3


# Precision ############################################################################


def _np_state_grads(model, x, h):
    """Local PC state gradients in float64 NumPy, from the weights and the given states."""
    W = [np.asarray(pcx.get(layer.nn.weight), np.float64) for layer in model.layers]
    b = [np.asarray(pcx.get(layer.nn.bias), np.float64) for layer in model.layers]
    xs = [np.asarray(x, np.float64), *[np.asarray(v, np.float64) for v in h]]
    eps = [xs[i + 1] - (xs[i] @ W[i].T + b[i]) for i in range(len(h))]  # linear activation
    eps.append(np.asarray(model.out.h.get(), np.float64) - (xs[-1] @ W[-1].T + b[-1]))
    return [eps[i] - eps[i + 1] @ W[i + 1] for i in range(len(h))]


def test_error_vodes_keep_the_gradient_when_the_error_is_small_relative_to_the_value():
    """|e| ~ 1e-3 |h|: a Vode forms e = h - u in float32 and loses a relative
    eps*|h|/|e| ~ 1e-4 of it per element; an ErrorVode reads the stored e.

    The reference uses the exact errors the ErrorVode stores, so it is the gradient
    at the ErrorVode's state; the Vode network is given the closest float32 states.
    """
    x, y = data(scale=1e3)
    plain, err = MLP(pxc.Vode, lambda a: a), MLP(pxc.ErrorVode, lambda a: a)
    init(plain, x, y)
    init(err, x, y)

    keys = jax.random.split(jax.random.PRNGKey(3), len(err.nodes))
    for k, n in zip(keys, err.nodes, strict=True):
        n.e.set(0.1 * jax.random.normal(k, n.e.get().shape))

    # Exact float64 values of the ErrorVode states: h = u + e with u from the float64 forward pass.
    W = [np.asarray(pcx.get(layer.nn.weight), np.float64) for layer in err.layers]
    bias = [np.asarray(pcx.get(layer.nn.bias), np.float64) for layer in err.layers]
    h64, prev = [], np.asarray(x, np.float64)
    for i, n in enumerate(err.nodes):
        prev = prev @ W[i].T + bias[i] + np.asarray(n.e.get(), np.float64)
        h64.append(prev)
    for pn, h in zip(plain.nodes, h64, strict=True):
        pn.h.set(jnp.asarray(h, jnp.float32))
    assert min(float(np.abs(h).mean()) for h in h64) > 100.0, "the states are not large relative to the errors"

    ref = _np_state_grads(err, x, h64)
    _, g_err = state_grads(err, x)
    _, g_plain = state_grads(plain, x)

    def rel(g, r):
        return float(np.linalg.norm(np.asarray(g, np.float64) - r) / np.linalg.norm(r))

    err_rel = max(rel(gn.e.get(), r) for gn, r in zip(g_err.nodes, ref, strict=True))
    plain_rel = max(rel(gn.h.get(), r) for gn, r in zip(g_plain.nodes, ref, strict=True))

    # Measured: ErrorVode 6e-8 (float32 resolution of e itself), Vode 2e-4.
    assert err_rel < 1e-6, f"ErrorVode gradient off by {err_rel:.2e}"
    assert plain_rel > 100 * err_rel, f"Vode {plain_rel:.2e} vs ErrorVode {err_rel:.2e}"


def test_under_init_the_weight_gradient_is_backprop():
    """At forward initialisation e = 0 and the energy is the output loss; its weight gradient must reach every
    layer as for a Vode network (the stop-gradient only applies outside STATUS.INIT)."""
    x, y = data()
    plain, err = MLP(pxc.Vode, jnp.tanh), MLP(pxc.ErrorVode, jnp.tanh)

    def bp_grads(model):
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to((False, True)), has_aux=True)(energy_y)(
                x, y, model=model
            )
        return [jnp.asarray(pcx.get(layer.nn.weight)) for layer in g["model"].layers]

    for wp, we in zip(bp_grads(plain), bp_grads(err), strict=True):
        assert float(jnp.linalg.norm(wp)) > 0
        assert_allclose(we, wp, rtol=1e-5, atol=1e-6)


@pxf.vmap(VMASK, in_axes=(0, 0), out_axes=(None, None), axis_name="batch")
def energy_y(x, y, *, model):
    model(x, y)
    per_node = jnp.stack([n.energy() for n in [*model.nodes, model.out]])
    per_node = jax.lax.psum(per_node, "batch")
    return per_node.sum(), per_node
