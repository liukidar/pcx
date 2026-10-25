"""`Cached`: a linear layer whose output is kept between inference steps.

The contract is that a network with cached layers computes exactly what the same
network without them computes -- energies, local PC gradients (states and weights)
and, with `apply_h_updates`, the inference trajectories -- while never evaluating
the layer's forward matmul after the forward initialisation. The oracles are the
uncached Vode and ErrorVode networks.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
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
    """x -> [layer -> act -> node] x 2 -> layer -> clamped output Vode."""

    def __init__(self, node: type[pxc.Vode], act: Callable[[jax.Array], jax.Array], cached: bool, seed: int = 0):
        super().__init__()
        rkg = pcx.RandomKeyGenerator(seed)
        self.act = pcx.static(act)
        layers = [pxnn.Linear(DIMS[i], DIMS[i + 1], rkg=rkg) for i in range(len(DIMS) - 1)]
        self.layers: list[Any] = [pxc.Cached(layer) if cached else layer for layer in layers]
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


VMASK = pxu.M(pxc.VodeParam | pxc.VodeParam.Cache | pxc.LayerCache).to((None, 0))


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
    return [jnp.asarray(pcx.get(_inner(layer).nn.weight)) for layer in g["model"].layers]


def _inner(layer):
    return layer.layer if isinstance(layer, pxc.Cached) else layer


def data():
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    return jax.random.normal(k1, (B, DIMS[0])), jax.random.normal(k2, (B, DIMS[-1]))


def pair(node=pxc.ErrorVode, act=jnp.tanh):
    """The same network (weights, states) with plain and with cached layers, at a perturbed state."""
    x, y = data()
    plain, cached = MLP(node, act, cached=False), MLP(node, act, cached=True)
    init(plain, x, y)
    init(cached, x, y)
    keys = jax.random.split(jax.random.PRNGKey(2), len(plain.nodes))
    for k, pn, cn in zip(keys, plain.nodes, cached.nodes, strict=True):
        eps = 0.3 * jax.random.normal(k, pn._stored().get().shape)
        if node is pxc.ErrorVode:
            pn.e.set(eps)
            cn.e.set(eps)
        else:
            pn.h.set(pn.h.get() + eps)
            cn.h.set(cn.h.get() + eps)
    # The caches hold z at the INIT states: refill them at the perturbed state.
    _refill(cached, x)
    return x, y, plain, cached


def vode_oracle(model, x, y):
    """A plain Vode network with the same weights, at the same values h as `model`."""
    oracle = MLP(pxc.Vode, model.act.get(), cached=False)
    init(oracle, x, y)
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        forward(x, None, model=model)
        for on, mn in zip(oracle.nodes, model.nodes, strict=True):
            on.h.set(jnp.asarray(mn.get("h")))
    return oracle


def values(model, x):
    """h of every hidden node, recomputed by a forward pass from the current states."""
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        forward(x, None, model=model)
        return [jnp.asarray(n.get("h")) for n in model.nodes]


def _refill(model, x):
    """Recompute every layer cache from the current states (an uncached forward pass)."""
    model.clear_params(pxc.LayerCache)
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        forward(x, None, model=model)


# Structure ############################################################################


def test_init_fills_the_cache_and_later_calls_return_it():
    x, y = data()
    model = MLP(pxc.ErrorVode, jnp.tanh, cached=True)
    init(model, x, y)
    zs = [layer.cache["z"] for layer in model.layers]
    assert all(z.shape[0] == B for z in zs)

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        forward(x, None, model=model)
    for layer, z in zip(model.layers, zs, strict=True):
        assert layer.cache["z"] is z or bool(jnp.array_equal(layer.cache["z"], z))


def test_clearing_vode_caches_keeps_the_layer_cache():
    x, y = data()
    model = MLP(pxc.ErrorVode, jnp.tanh, cached=True)
    init(model, x, y)
    model.clear_params(pxc.VodeParam.Cache)
    assert all(layer.cache.get("z") is not None for layer in model.layers)
    model.clear_params(pxc.LayerCache)
    assert all(layer.cache.get() is None for layer in model.layers)


# Equivalence with the uncached network ################################################


@pytest.mark.parametrize("node", [pxc.Vode, pxc.ErrorVode])
def test_energies_and_state_gradients_match(node):
    x, _, plain, cached = pair(node)
    e_plain, g_plain = state_grads(plain, x)
    e_cached, g_cached = state_grads(cached, x)
    assert_allclose(e_cached, e_plain, rtol=1e-5, atol=1e-6)
    for gp, gc in zip(g_plain.nodes, g_cached.nodes, strict=True):
        assert_allclose(gc._stored().get(), gp._stored().get(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("node", [pxc.Vode, pxc.ErrorVode])
def test_weight_gradients_match(node):
    x, _, plain, cached = pair(node)
    for wp, wc in zip(weight_grads(plain, x), weight_grads(cached, x), strict=True):
        assert_allclose(wc, wp, rtol=1e-5, atol=1e-6)


def test_the_cached_forward_evaluates_no_matmul():
    """After INIT, the forward pass of the cached network contains no dot_general."""
    x, y = data()
    model = MLP(pxc.ErrorVode, jnp.tanh, cached=True)
    init(model, x, y)
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        text = str(jax.make_jaxpr(lambda x_: forward(x_, None, model=model))(x))
    assert "dot_general" not in text


def _run(model, x, T, lr=0.05, momentum=0.5, h_steps=True):
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

    optim_h = pxu.Optim(lambda: optax.sgd(lr, momentum=momentum))
    optim_h.init(_state_mask()(model))
    return infer(x, model=model, optim_h=optim_h)


@pytest.mark.parametrize("node", [pxc.Vode, pxc.ErrorVode])
@pytest.mark.parametrize("lr, rtol", [(0.05, 1e-3), (0.005, 1e-5)])
def test_inference_trajectory_matches_a_vode_network(node, lr, rtol):
    """SGD with momentum under jit: the cached network follows the plain Vode network's energy trajectory and ends
    at the same values. With ErrorVodes the h-step is a second-order Taylor step, so the trajectory differs from the
    Vode one by O(|d|^3) per step: the large-step case (errors at 30% of |h|, lr 0.05, momentum 0.5) shows it at
    the 1e-4 level, the small-step case is at float32 level."""
    x, y, _, cached = pair(node)
    oracle = vode_oracle(cached, x, y)
    traj_oracle = _run(oracle, x, 20, lr=lr, h_steps=False)
    traj_cached = _run(cached, x, 20, lr=lr)
    assert_allclose(traj_cached, traj_oracle, rtol=rtol, atol=1e-6)
    for h_c, on in zip(values(cached, x), oracle.nodes, strict=True):
        assert_allclose(h_c, on.h.get(), rtol=rtol, atol=1e-5)


@pytest.mark.parametrize("node", [pxc.Vode, pxc.ErrorVode])
def test_incremental_caches_equal_a_fresh_forward(node):
    """z <- z + W d keeps each cache equal to W h + b at the current states. With ErrorVodes the Taylor remainder
    of the error update enters z at O(|d|^3) per step: exact for Vodes, and at float32 level for small steps."""
    x, _, _, cached = pair(node)
    _run(cached, x, 20, lr=0.005, momentum=0.0)
    zs = [layer.cache["z"] for layer in cached.layers]
    _refill(cached, x)
    for z_inc, layer in zip(zs, cached.layers, strict=True):
        assert_allclose(z_inc, layer.cache["z"], rtol=1e-5, atol=1e-6)


def test_apply_h_updates_is_exact_for_a_linear_network():
    x, _, _, cached = pair(pxc.ErrorVode, act=lambda a: a)
    before = values(cached, x)
    _, g = state_grads(cached, x)
    updates = jax.tree.map(lambda a: -0.5 * a, g)
    with pxu.step(cached, clear_params=pxc.VodeParam.Cache):
        pxc.apply_h_updates(forward, updates, x, None, model=cached, order=1)
    for h1, h0, gn in zip(values(cached, x), before, g.nodes, strict=True):
        assert_allclose(h1, h0 - 0.5 * gn.e.get(), rtol=1e-5, atol=1e-6)


# Convolutions ########################################################################


class CNN(pxc.EnergyModule):
    """x (2, 6, 6) -> [Conv -> tanh -> node] x 2 -> flatten -> Linear -> clamped output Vode."""

    def __init__(self, node: type[pxc.Vode], cached: bool, seed: int = 0):
        super().__init__()
        rkg = pcx.RandomKeyGenerator(seed)
        layers = [
            pxnn.Conv2d(2, 3, 3, padding=1, rkg=rkg),
            pxnn.Conv2d(3, 2, 3, padding=1, rkg=rkg),
            pxnn.Linear(2 * 6 * 6, DIMS[-1], rkg=rkg),
        ]
        self.layers: list[Any] = [pxc.Cached(layer) if cached else layer for layer in layers]
        self.nodes: list[Any] = [node() for _ in range(2)]
        self.out = pxc.Vode()
        self.out.h.frozen = True

    def __call__(self, x, y):
        for layer, node in zip(self.layers[:-1], self.nodes, strict=True):
            x = node(jnp.tanh(layer(x)))
        self.out(self.layers[-1](x.reshape(-1)))
        if y is not None:
            self.out.set("h", y)
        return self.out.get("u")


def cnn_pair(node):
    k1, k2 = jax.random.split(jax.random.PRNGKey(4))
    x, y = jax.random.normal(k1, (B, 2, 6, 6)), jax.random.normal(k2, (B, DIMS[-1]))
    plain, cached = CNN(node, cached=False), CNN(node, cached=True)
    init(plain, x, y)
    init(cached, x, y)
    keys = jax.random.split(jax.random.PRNGKey(5), len(plain.nodes))
    for k, pn, cn in zip(keys, plain.nodes, cached.nodes, strict=True):
        eps = 0.3 * jax.random.normal(k, pn._stored().get().shape)
        if node is pxc.ErrorVode:
            pn.e.set(eps)
            cn.e.set(eps)
        else:
            pn.h.set(pn.h.get() + eps)
            cn.h.set(cn.h.get() + eps)
    _refill(cached, x)
    return x, y, plain, cached


@pytest.mark.parametrize("node", [pxc.Vode, pxc.ErrorVode])
def test_conv_gradients_match(node):
    """The tangent rule is generic: for a convolution JAX transposes it to the usual conv gradients."""
    x, _, plain, cached = cnn_pair(node)
    e_plain, g_plain = state_grads(plain, x)
    e_cached, g_cached = state_grads(cached, x)
    assert_allclose(e_cached, e_plain, rtol=1e-5, atol=1e-6)
    for gp, gc in zip(g_plain.nodes, g_cached.nodes, strict=True):
        assert_allclose(gc._stored().get(), gp._stored().get(), rtol=1e-5, atol=1e-6)
    for wp, wc in zip(weight_grads(plain, x), weight_grads(cached, x), strict=True):
        assert_allclose(wc, wp, rtol=1e-5, atol=1e-6)


def test_conv_cached_forward_evaluates_no_convolution():
    x, y, _, cached = cnn_pair(pxc.ErrorVode)
    with pxu.step(cached, clear_params=pxc.VodeParam.Cache):
        text = str(jax.make_jaxpr(lambda x_: forward(x_, None, model=cached))(x))
    pcx.RKG.seed(0)
    assert "conv_general_dilated" not in text and "dot_general" not in text


def test_conv_trajectory_matches_a_vode_network():
    x, y, _, cached = cnn_pair(pxc.ErrorVode)
    oracle = CNN(pxc.Vode, cached=False)
    init(oracle, x, y)
    for on, h in zip(oracle.nodes, values(cached, x), strict=True):
        on.h.set(h)
    traj_oracle = _run(oracle, x, 20, lr=0.005, h_steps=False)
    traj_cached = _run(cached, x, 20, lr=0.005)
    assert_allclose(traj_cached, traj_oracle, rtol=1e-5, atol=1e-6)
