from typing import Callable
import random

import jax
import numpy as np
import optax
from sklearn.datasets import make_moons

import pcax as px
import pcax.nn as pxnn
import pcax.functional as pxf
import pcax.utils as pxu


def get_data(nm_elements, batch_size):
    X, y = make_moons(n_samples=batch_size * (nm_elements // batch_size), noise=0.2, random_state=42)

    train_dl = list(zip(X.reshape(-1, batch_size, 2), y.reshape(-1, batch_size)))

    X_test, y_test = make_moons(n_samples=batch_size * (nm_elements // batch_size) // 2, noise=0.2, random_state=0)
    test_dl = tuple(zip(X_test.reshape(-1, batch_size, 2), y_test.reshape(-1, batch_size)))

    return train_dl, test_dl


def ce_loss(output, one_hot_label):
    return -one_hot_label * jax.nn.log_softmax(output)


class Model(px.Module):
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

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.act_fn(layer(x))

        x = self.layers[-1](x)

        return x


@pxf.vmap({"model": None}, in_axes=0, out_axes=0)
def forward(x, *, model: Model):
    return model(x)


@pxf.vmap({"model": None}, in_axes=(0, 0), out_axes=(None, 0), axis_name="batch")
def loss(x, y, *, model: Model):
    y_ = model(x)
    return jax.lax.pmean(ce_loss(y_, y).sum(), "batch"), y_


@pxf.jit()
def train_on_batch(x: jax.Array, y: jax.Array, *, model: Model, optim_w: pxu.Optim):
    model.train()

    with pxu.step(model):
        (e, y_), g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(loss)(x, y, model=model)
    optim_w.step(model, g["model"])


@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: Model):
    model.eval()

    with pxu.step(model):
        y_ = forward(x, model=model).argmax(axis=-1)

    return (y_ == y).mean(), y_


def train(dl, *, model: Model, optim_w: pxu.Optim):
    for x, y in dl:
        train_on_batch(x, jax.nn.one_hot(y, 2), model=model, optim_w=optim_w)


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

with pxu.step(model):
    optim_w = pxu.Optim(optax.adamw(1e-2), pxu.Mask(pxnn.LayerParam)(model))

for e in range(nm_epochs):
    random.shuffle(train_dl)
    train(train_dl, model=model, optim_w=optim_w)
    a, y = eval(test_dl, model=model)

    print(f"Epoch {e + 1}/{nm_epochs} - Test Accuracy: {a * 100:.2f}%")
