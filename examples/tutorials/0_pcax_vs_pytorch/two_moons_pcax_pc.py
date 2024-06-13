from typing import Callable
import random

import jax
import numpy as np
import optax
from sklearn.datasets import make_moons

import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.functional as pxf
import pcax.utils as pxu


def get_data(nm_elements, batch_size):
    X, y = make_moons(n_samples=batch_size * (nm_elements // batch_size), noise=0.2, random_state=42)

    train_dl = list(zip(X.reshape(-1, batch_size, 2), y.reshape(-1, batch_size)))

    X_test, y_test = make_moons(n_samples=batch_size * (nm_elements // batch_size) // 2, noise=0.2, random_state=0)
    test_dl = tuple(zip(X_test.reshape(-1, batch_size, 2), y_test.reshape(-1, batch_size)))

    return train_dl, test_dl


class Model(pxc.EnergyModule):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, nm_layers: int, act_fn: Callable[[jax.Array], jax.Array]
    ) -> None:
        super().__init__()

        self.act_fn = px.static(act_fn)

        self.layers = (
            [pxnn.Linear(input_dim, hidden_dim)]
            + [pxnn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 2)]
            + [pxnn.Linear(hidden_dim, output_dim)]
        )

        self.vodes = [pxc.Vode((hidden_dim,)) for _ in range(nm_layers - 1)] + [pxc.Vode((output_dim,), pxc.ce_energy)]

        self.vodes[-1].h.frozen = True

    def __call__(self, x, y):
        for v, l in zip(self.vodes[:-1], self.layers[:-1]):
            x = v(self.act_fn(l(x)))

        x = self.vodes[-1](self.layers[-1](x))

        if y is not None:
            self.vodes[-1].set("h", y)

        return self.vodes[-1].get("u")


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: Model):
    return model(x, y)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: Model):
    y_ = model(x, None)
    return jax.lax.pmean(model.energy().sum(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch(T: int, x: jax.Array, y: jax.Array, *, model: Model, optim_w: pxu.Optim, optim_h: pxu.Optim):
    model.train()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)

    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, y_), g = pxf.value_and_grad(
                pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]), has_aux=True
            )(energy)(x, model=model)

        optim_h.step(model, g["model"])

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, y_), g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(energy)(x, model=model)
    optim_w.step(model, g["model"])


@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: Model):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model).argmax(axis=-1)

    return (y_ == y).mean(), y_


def train(dl, T, *, model: Model, optim_w: pxu.Optim, optim_h: pxu.Optim):
    for x, y in dl:
        train_on_batch(T, x, jax.nn.one_hot(y, 2), model=model, optim_w=optim_w, optim_h=optim_h)


def eval(dl, *, model: Model):
    acc = []
    ys_ = []

    for x, y in dl:
        a, y_ = eval_on_batch(x, y, model=model)
        acc.append(a)
        ys_.append(y_)

    return np.mean(acc), np.concatenate(ys_)


batch_size = 32
nm_elements = 1024
nm_epochs = 256 // (nm_elements // batch_size)

model = Model(input_dim=2, hidden_dim=32, output_dim=2, nm_layers=3, act_fn=jax.nn.leaky_relu)
train_dl, test_dl = get_data(nm_elements, batch_size)

with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
    forward(jax.numpy.zeros((batch_size, 2)), None, model=model)

    optim_h = pxu.Optim(optax.sgd(1e-2 * batch_size), pxu.Mask(pxc.VodeParam)(model))
    optim_w = pxu.Optim(optax.adamw(1e-2), pxu.Mask(pxnn.LayerParam)(model))

for e in range(nm_epochs):
    random.shuffle(train_dl)
    train(train_dl, T=8, model=model, optim_w=optim_w, optim_h=optim_h)
    a, y = eval(test_dl, model=model)

    print(f"Epoch {e + 1}/{nm_epochs} - Test Accuracy: {a * 100:.2f}%")
