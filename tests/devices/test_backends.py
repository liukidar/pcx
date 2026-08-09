"""End-to-end smoke tests per accelerator backend.

These answer one question per backend: does a real training loop run and make
progress? They deliberately do not localise faults — Tiers 1-3 do that. Their
job is to catch total breakage on hardware that CI cannot reach.

Every test here is marked ``device`` and is excluded from the default run and
from GitHub Actions, which has no GPU. Run them on a workstation with::

    just test-devices

A backend that is not present is skipped, not failed, so the same command is
meaningful on a CPU-only laptop and on a CUDA box.

NOTE: ``conftest.py`` pins ``JAX_PLATFORMS=cpu`` before jax is imported, so the
accelerator backends are only visible when that is overridden::

    JAX_PLATFORMS='' just test-devices
"""

import jax
import jax.numpy as jnp
import optax
import pytest

import pcx
import pcx.functional as pxf
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
import pcx.utils as pxu

pytestmark = pytest.mark.device


def available_backends() -> set[str]:
    """Platforms jax can actually reach in this process."""
    try:
        return {d.platform for d in jax.devices()}
    except RuntimeError:
        return set()


def requires(platform: str):
    return pytest.mark.skipif(
        platform not in available_backends(),
        reason=f"no {platform} backend available (saw {sorted(available_backends()) or 'none'})",
    )


class TinyPC(pxc.EnergyModule):
    """The smallest supervised model exercising layers, Vodes and energy together.

    The output node is clamped to the target, so the energy is the squared
    prediction error. Without clamping, forward initialisation leaves `h == u`
    and the energy is identically zero — correct, but nothing to learn from.
    """

    def __init__(self, rkg: pcx.RandomKeyGenerator):
        super().__init__()
        self.layer = pxnn.Linear(4, 3, rkg=rkg)
        self.vode = pxc.Vode()
        self.vode.h.frozen = True

    def __call__(self, x, y=None):
        u = self.vode(jax.nn.tanh(self.layer(x)))
        if y is not None:
            self.vode.set("h", y)
        return u


def loss_fn(x, y, *, model):
    model(x, y)
    return model.energy().sum()


def run_training_step(platform: str):
    """One inference + weight update on `platform`. Returns (energy, moved)."""
    device = jax.devices(platform)[0]
    model = TinyPC(pcx.RandomKeyGenerator(seed=0))
    x = jax.device_put(jnp.ones((4,)), device)
    y = jax.device_put(jnp.array([1.0, -1.0, 0.5]), device)

    # Forward-initialise: under STATUS.INIT the default ruleset copies the
    # incoming activation `u` into the node value `h`. Without this pass `h` is
    # None and the energy is undefined.
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        model(x, y)

    before = jnp.asarray(pcx.get(model.layer.nn.weight)).copy()
    optim = pxu.Optim(lambda: optax.sgd(1e-1), pxu.M(pxnn.LayerParam)(model))

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        energy, grads = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to((False, True)))(loss_fn)(x, y, model=model)

    optim.step(pxu.M(pxnn.LayerParam)(model), grads["model"])

    after = jnp.asarray(pcx.get(model.layer.nn.weight))
    return float(energy), not jnp.array_equal(before, after)


@requires("cpu")
def test_training_step_runs_on_cpu():
    energy, moved = run_training_step("cpu")

    assert jnp.isfinite(energy), f"energy is not finite: {energy}"
    assert moved, "weights did not change after an optimiser step"


@requires("gpu")
def test_training_step_runs_on_cuda():
    """NVIDIA GPU. Needs `pip install -U 'jax[cuda12]'` — Linux only; JAX ships
    no CUDA wheels for native Windows."""
    energy, moved = run_training_step("gpu")

    assert jnp.isfinite(energy), f"energy is not finite: {energy}"
    assert moved, "weights did not change after an optimiser step"


@requires("METAL")
def test_training_step_runs_on_apple_metal():
    """Apple Silicon via the experimental `jax-metal` plugin."""
    energy, moved = run_training_step("METAL")

    assert jnp.isfinite(energy), f"energy is not finite: {energy}"
    assert moved, "weights did not change after an optimiser step"


@pytest.mark.parametrize("platform", ["gpu", "METAL"])
def test_accelerator_agrees_with_cpu(platform: str):
    """An accelerator that returns different numbers from the CPU reference is
    the failure mode that silently corrupts a research result."""
    if platform not in available_backends():
        pytest.skip(f"no {platform} backend available")

    cpu_energy, _ = run_training_step("cpu")
    acc_energy, _ = run_training_step(platform)

    assert jnp.allclose(cpu_energy, acc_energy, rtol=1e-4, atol=1e-5), (
        f"{platform} energy {acc_energy} disagrees with CPU {cpu_energy}"
    )
