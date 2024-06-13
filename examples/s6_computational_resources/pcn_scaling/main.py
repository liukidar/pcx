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


STATUS_FORWARD = "forward"


class pMLP(pxc.EnergyModule):
    def __init__(self, dim, L, act_fn: Callable[[jax.Array], jax.Array]) -> None:
        super().__init__()

        self.act_fn = px.static(act_fn)
        self.dim = px.static(dim)
        self.pLayer = pxnn.Linear(dim, dim * L)
        
        self.pLayer.nn.weight.set(self.pLayer.nn.weight.reshape(L, dim, dim))
        self.pLayer.nn.bias.set(self.pLayer.nn.bias.reshape(L, dim, dim))

        self.vode = pxc.Vode(
            (dim * (L + 1),),
            ruleset={
                pxc.STATUS.INIT: ("h, u <- u:to_zero",),
                STATUS_FORWARD: ("h -> u",)
            },
            tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros(n.shape)}
        )
    
    def energy(self):
        return jax.numpy.square(self.vode.get("u") - self.vode.get("h")[self.dim.get():]).sum()

    def __call__(self, x: jax.Array, y: jax.Array):
        self.vode.h.set(self.vode.h.at[:self.dim.get()].set(x))
        self.vode(self.vode.h[:-self.dim.get()])

        if y is not None:
            self.vode.h.set(self.vode.h.at[-self.dim.get():].set(y))


class MLP(pxc.EnergyModule):
    def __init__(self, dim, L, act_fn: Callable[[jax.Array], jax.Array]) -> None:
        super().__init__()

        self.act_fn = px.static(act_fn)

        self.layers = [
            (pxnn.Linear(dim, dim), self.act_fn) for _ in range(L)
        ]
        
        self.vodes = [
            pxc.Vode(
                (dim,),
                ruleset={
                    pxc.STATUS.INIT: ("h, u <- u:to_zero",),
                    STATUS_FORWARD: ("h -> u",)
                },
                tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros(n.shape)}
            ) for _ in range(L)
        ]

        self.vodes[-1].h.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array):
        for i, (block, node) in enumerate(zip(self.layers, self.vodes)):
            for layer in block:
                x = layer(x)
            x = node(x)

        if y is not None:
            self.vodes[-1].set("h", y)

        return self.vodes[-1].get("u")


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: MLP):
    return model(x, y)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=None, axis_name="batch")
def energy(x, *, model: MLP):
    model(x, None)
    return jax.lax.pmean(model.energy().sum(), "batch")


@pxf.jit(static_argnums=0)
def train_on_batch(
    T: int, x: jax.Array, y: jax.Array, *, model: MLP, optim_w: pxu.Optim, optim_h: pxu.Optim
):
    model.train()
    print("Running...")

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


def train(dl, T, *, model: MLP, optim_w: pxu.Optim, optim_h: pxu.Optim):
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
    model = MLP(dim, L, act_fn=jax.nn.relu)
    
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
    timings = {"B": [], "D": [], "L": [], "T": []}
    BASE_B = 32
    BASE_D = 64
    BASE_L = 2
    BASE_T = 8
    
    MULS = [1, 2, 3, 4, 5, 6]
    
    for mul in MULS:
        _clock = main(BASE_B * mul, BASE_D, BASE_L, BASE_T, mul)
        timings["B"].append(_clock)
    for mul in MULS:
        _clock = main(BASE_B, BASE_D * mul, BASE_L, BASE_T, mul)
        timings["D"].append(_clock)
    for mul in MULS:
        _clock = main(BASE_B, BASE_D, BASE_L* mul, BASE_T, mul)
        timings["L"].append(_clock)
    for mul in MULS:
        _clock = main(BASE_B, BASE_D, BASE_L, BASE_T * mul, mul)
        timings["T"].append(_clock)
    
    with open("main_timings.json", "w") as f:
        json.dump(timings, f, indent=4)