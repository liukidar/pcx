"""`pxf.vmap` over a batch sharded on an explicit mesh.

Since jax 0.9.0 `jax.make_mesh` builds meshes of `AxisType.Explicit` by default,
where an array's sharding is part of its type. `jax.vmap` then requires every
input mapped over the same axis to be sharded the same way along it. `Vmap._t`
splits the global RKG key into one key per lane and maps it alongside the batch,
so an unsharded key next to a batch sharded on `P("data")` is rejected before the
function runs. This is the default data-parallel recipe, so the failure blocks
multi-device training outright.

The oracle is the same computation on an unsharded batch: sharding changes where
the numbers live, never what they are, so random draws, energies and updated
weights must agree.

Explicit sharding needs more than one device, and the device count is fixed when
jax initialises its backend. The checks therefore run in a subprocess with
`--xla_force_host_platform_device_count=2`.
"""

import json
import os
import subprocess
import sys

import jax
import pytest

pytestmark = pytest.mark.skipif(
    not (hasattr(jax.sharding, "AxisType") and hasattr(jax.sharding, "reshard")),
    reason=f"jax {jax.__version__} predates explicit sharding",
)

SCRIPT = r"""
import json

import jax
import jax.numpy as jnp
import optax
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

import pcx
import pcx.functional as pxf
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
import pcx.utils as pxu

assert jax.device_count() == 2, jax.devices()
mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Explicit,))
set_mesh = jax.set_mesh if hasattr(jax, "set_mesh") else jax.sharding.use_mesh
out = {}


def draw(x):
    return x + jax.random.normal(pcx.RKG(), ())


class Model(pxc.EnergyModule):
    def __init__(self):
        super().__init__()
        self.l1 = pxnn.Linear(3, 4)
        self.l2 = pxnn.Linear(4, 2)
        self.v1 = pxc.Vode()
        self.v2 = pxc.Vode()
        self.v2.h.frozen = True

    def __call__(self, x, y):
        x = self.v1(jax.nn.tanh(self.l1(x)))
        x = self.v2(self.l2(x))
        if y is not None:
            self.v2.set("h", y)
        return self.v2.get("u")


VMASK = pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0))


@pxf.vmap(VMASK, in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model):
    return model(x, y)


# The tutorials reduce with `jax.lax.psum(..., "batch")` inside the vmap. Under `jit` on an explicit mesh that
# fails inside jax itself (0.11 and 0.12, independent of pcx), so the batch is reduced outside the vmap instead.
@pxf.vmap(VMASK, in_axes=(0, 0), out_axes=(0, 0))
def energy_per_sample(x, y, *, model):
    y_ = model(x, y)
    return model.energy(), y_


def energy(x, y, *, model):
    e, y_ = energy_per_sample(x, y, model=model)
    return e.sum(), y_


@pxf.jit()
def train_on_batch(x, y, *, model, optim_w, optim_h):
    with pxu.step(model, (pxc.STATUS.INIT, None), clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)

    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    for _ in range(3):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, _), g = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(
                energy
            )(x, None, model=model)
        optim_h.step(model, g["model"])

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, _), g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)(
            x, None, model=model
        )
    optim_w.step(model, g["model"])
    return e


def train(x, y):
    pcx.RKG.seed(0)
    model = Model()
    with pxu.step(model, (pxc.STATUS.INIT, None), clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)
    optim_w = pxu.Optim(lambda: optax.sgd(1e-1), pxu.M(pxnn.LayerParam)(model))
    optim_h = pxu.Optim(lambda: optax.sgd(1e-1))
    e = train_on_batch(x, y, model=model, optim_w=optim_w, optim_h=optim_h)
    return float(e), [float(w) for w in jnp.ravel(pcx.get(model.l1.nn.weight))]


x = jnp.linspace(-1.0, 1.0, 24).reshape(8, 3)
y = jnp.linspace(-1.0, 1.0, 16).reshape(8, 2)

with set_mesh(mesh):
    xs = jax.device_put(x, NamedSharding(mesh, P("data")))
    ys = jax.device_put(y, NamedSharding(mesh, P("data")))
    out["batch_axis"] = jax.typeof(xs).sharding.spec[0]

    for name, wrap in [("eager", lambda f: f), ("jit", lambda f: pxf.jit()(f))]:
        g = wrap(pxf.vmap(in_axes=(0,), out_axes=0)(draw))
        pcx.RKG.seed(0)
        sharded = g(xs)
        out[f"draw_{name}_key_shape"] = list(pcx.RKG.key.get().shape)
        pcx.RKG.seed(0)
        plain = g(x)
        out[f"draw_{name}_equal"] = bool(jnp.array_equal(sharded, plain))
        out[f"draw_{name}_distinct"] = len(set((sharded - x)[:, 0].tolist())) == 8

    out["train_sharded"] = train(xs, ys)
    out["train_plain"] = train(x, y)

print("RESULT " + json.dumps(out))
"""


@pytest.fixture(scope="module")
def result() -> dict:
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": " ".join([os.environ.get("XLA_FLAGS", ""), "--xla_force_host_platform_device_count=2"]),
    }
    proc = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"the 2-device script failed:\n{proc.stderr[-4000:]}"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line.removeprefix("RESULT "))


def test_the_batch_is_explicitly_sharded(result):
    """Guards the setup: without an explicit spec on the batch the other tests
    would pass without exercising the failure."""
    assert result["batch_axis"] == "data", result["batch_axis"]


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_vmap_over_a_sharded_batch_draws_the_same_keys_as_unsharded(result, mode):
    """Each lane must get the same key it would get on one device, and the lanes
    must still be independent."""
    assert result[f"draw_{mode}_equal"], "sharded and unsharded draws differ"
    assert result[f"draw_{mode}_distinct"], "lanes drew duplicate samples"


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_vmap_over_a_sharded_batch_restores_a_single_key(result, mode):
    assert result[f"draw_{mode}_key_shape"] == [2]


def test_a_training_step_on_a_sharded_batch_matches_one_device(result):
    """Init, inference and a weight update, as in the tutorials, data-parallel
    over two devices. Sharding must not change the energy or the update."""
    e_sharded, w_sharded = result["train_sharded"]
    e_plain, w_plain = result["train_plain"]

    assert e_sharded == pytest.approx(e_plain, rel=1e-5, abs=1e-6)
    assert w_sharded == pytest.approx(w_plain, rel=1e-5, abs=1e-6)
