from typing import Callable
import numpy as np


# Core dependencies
import jax
import jax.numpy as jnp
import optax

# pcax
import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.utils as pxu
import pcax.functional as pxf

import clock
import json
import time


class pMLP(pxc.EnergyModule):
    def __init__(self, dim, L, act_fn: Callable[[jax.Array], jax.Array]) -> None:
        super().__init__()

        self.act_fn = px.static(act_fn)
        self.dim = px.static(dim)
        self.pLayer = pxnn.Linear(dim, dim * L)
        
        self.pLayer.nn.weight.set(self.pLayer.nn.weight.reshape(L, dim, dim))
        self.pLayer.nn.bias.set(self.pLayer.nn.bias.reshape(L, dim))
        
        def to_zero(n, k, v, rkg):
            h = jnp.zeros(n.shape).at[:self.dim.get()].set(v)
            return h

        self.vode = pxc.Vode(
            (dim * (L + 1),),
            ruleset={
                pxc.STATUS.INIT: ("h <- h:to_zero",),
            },
            tforms={"to_zero": to_zero}
        )
    
    def energy(self):
        return jax.numpy.square(self.vode.get("u") - self.vode.get("h")[self.dim.get():]).sum()

    def __call__(self, x: jax.Array, y: jax.Array):
        @pxf.vmap(pxu.Mask(pxnn.LayerParam, (None, 0)), in_axes=0, out_axes=0, axis_name="params")
        def step(x, *, model):
            return model.act_fn(model.pLayer(x))
        
        if self.status == pxc.STATUS.INIT:
            self.vode.set("h", x)
        h = self.vode.get("h")[:-self.dim.get()]
        u = step(h.reshape(-1, self.dim.get()), model=self)
        self.vode(u.flatten())

        if y is not None:
            self.vode.h.set(self.vode.h.at[-self.dim.get():].set(y))


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: pMLP):
    return model(x, y)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=None, axis_name="batch")
def energy(x, *, model: pMLP):
    model(x, None)
    return jax.lax.pmean(model.energy(), "batch")


@pxf.jit(static_argnums=0)
def train_on_batch(
    T: int, x: jax.Array, y: jax.Array, *, model: pMLP, optim_w: pxu.Optim, optim_h: pxu.Optim
):
    model.train()

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)
    optim_h.init(pxu.Mask(pxc.VodeParam)(model))

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]), has_aux=False)(
                energy
            )(x, model=model)

        optim_h.step(model, g["model"], True)
    optim_h.clear()

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        l, g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=False)(energy)(x, model=model)
    optim_w.step(model, g["model"])
    
    return l


def train(dl, T, *, model: pMLP, optim_w: pxu.Optim, optim_h: pxu.Optim):
    for i, (x, y) in enumerate(dl):
        l = train_on_batch(
            T, x, y, model=model, optim_w=optim_w, optim_h=optim_h
        )
    return l


def main(
    batch_size: int,
    dim: int,
    L: int,
    T: int,
    mul: int
):
    nm_epochs = 6
    model = pMLP(dim, L, act_fn=jax.nn.relu)
    
    dummy_data = [(
        jax.random.normal(px.RKG(), (batch_size, dim)), jax.random.normal(px.RKG(), (batch_size, dim))
    )] * 1024

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(dummy_data[0][0], None, model=model)

        optim_h = pxu.Optim(
            optax.chain(
                optax.sgd(5e-2),
            )
        )
        optim_w = pxu.Optim(
            optax.adamw(1e-3),
            pxu.Mask(pxnn.LayerParam)(model),
        )

    train_c = clock.timed_fn(train, {"mul": mul})
    for e in range(nm_epochs):
        train_c(dummy_data, T=T, model=model, optim_w=optim_w, optim_h=optim_h)
    
        time.sleep(2.0)
    
    return train_c.save()


if __name__ == "__main__":
    timings = {"vL": []}
    BASE_B = 32
    BASE_D = 64
    BASE_L = 2
    BASE_T = 8
    
    MULS = [1, 2, 3, 4, 5, 6]
    
    for mul in MULS:
        _clock = main(BASE_B, BASE_D, BASE_L* mul, BASE_T, mul)
        timings["vL"].append(_clock)
    
    with open("pmain_timings.json", "w") as f:
        json.dump(timings, f, indent=4)