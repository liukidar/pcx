import jax
import optax
import pcax.interface as pxi
import numpy as np
from torchvision.datasets import MNIST
import timeit

import pcax as px
import pcax.nn as nn
from pcax.core import _

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

NM_LAYERS = 16


class Model(px.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nm_layers=NM_LAYERS) -> None:
        super().__init__()

        self.act_fn = jax.nn.tanh

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear_h = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers)]
        )
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.pc1 = px.Layer()
        self.pc1.x.frozen = 0
        self.pc_h = nn.ModuleList([px.Layer() for _ in range(nm_layers)])
        for i in range(nm_layers):
            self.pc_h[i].x.frozen = i + 1
        self.pc2 = px.Layer()

        self.pc2.x.frozen = NM_LAYERS + 1

    def __call__(self, x, t=None):
        x = self.pc1(self.act_fn(self.linear1(x)))["x"]

        for i in range(len(self.linear_h)):
            x = self.pc_h[i](self.act_fn(self.linear_h[i](x)))["x"]

        x = self.pc2(self.linear2(x))["x"]

        if t is not None:
            self.pc2["x"] = t

        return self.pc2["u"]


def one_hot(x, k):
    return np.array(x[:, None] == np.arange(k), dtype=np.float32)


class FlattenAndCast:
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=np.float32) / 255.0)


params = {
    "batch_size": 32,
    "x_learning_rate": 0.01,
    "w_learning_rate": 1e-3,
    "num_epochs": 4,
    "hidden_dim": 128,
    "input_dim": 28 * 28,
    "output_dim": 10,
    "seed": 0,
    "T": NM_LAYERS + 2,
}

train_dataset = MNIST(
    "/tmp/mnist/",
    transform=FlattenAndCast(),
    train=True,
)
train_dataloader = pxi.data.Dataloader(
    train_dataset,
    batch_size=params["batch_size"],
    num_workers=4,
    shuffle=False,
    persistent_workers=True,
    pin_memory=True,
)

test_dataset = MNIST(
    "/tmp/mnist/",
    transform=FlattenAndCast(),
    train=False,
)
test_dataloader = pxi.data.Dataloader(
    test_dataset,
    batch_size=params["batch_size"],
    num_workers=4,
    shuffle=False,
    persistent_workers=False,
    pin_memory=True,
)

model = Model(28 * 28, params["hidden_dim"], 10)


@px.vectorize(_(px.NodeVar), in_axis=(0, 0))
@px.bind(model)
def predict(x, t):
    return model(x, t)


@px.vectorize(_(px.NodeVar), in_axis=(0, None), out_axis=("sum",))
@px.bind(model)
def train_op(x, i):
    if i == 0:
        model.pc1(model.act_fn(model.linear1(x)))["x"]

        return model.pc1.energy
    elif i == NM_LAYERS + 1:
        model.pc2(model.linear2(model.pc_h[-1]["x"]))["x"]

        return model.pc2.energy
    else:
        x = model.pc_h[i - 2]["x"] if i > 1 else model.pc1["x"]
        model.pc_h[i - 1](model.act_fn(model.linear_h[i - 1](x)))["x"]

        return model.pc_h[i - 1].energy


@px.vectorize(_(px.NodeVar), in_axis=(0,), out_axis=("sum",))
@px.bind(model)
def train_op_w(x):
    model(x)
    return model.energy


train_x = px.gradvalues(
    _[_(px.NodeVar), -_(px.NodeVar)(frozen=NM_LAYERS + 1)],
)(train_op_w)
train_w = px.gradvalues(
    _[px.TrainVar, -_(px.NodeVar)],
)(train_op_w)


# dummy run to init the optimizer parameters
with px.eval(model):
    predict(np.zeros((params["batch_size"], 28 * 28)), (None,) * params["batch_size"])
    optim_x = px.Optim(
        optax.sgd(params["x_learning_rate"]), model.vars(_[_(px.NodeVar), -_(px.NodeVar)(frozen=NM_LAYERS + 1)])
    )
    optim_w = px.Optim(
        optax.adam(params["w_learning_rate"]),
        model.vars(_[px.TrainVar, -_(px.NodeVar)]),
    )


@px.jit()
@px.bind(model, optim_w=optim_w, optim_x=optim_x)
def train_on_batch(x, y):
    train_xs = (px.gradvalues(
        _(px.NodeVar)(frozen=0)
    )(train_op),) + tuple(
        px.gradvalues(
            _(_(px.NodeVar)(frozen=i), _(px.NodeVar)(frozen=i + 1))
        )(train_op) for i in range(0, NM_LAYERS)
    ) + (px.gradvalues(
        _(_(px.NodeVar)(frozen=NM_LAYERS))
    )(train_op),)

    with px.eval(model):
        predict(x, y)

        for i in range(params["T"]):
            with px.train(model):
                gs = tuple(
                    train_xs[i](x, i)[0] for i in range(NM_LAYERS + 2)
                )

                g = {}

                for i, v in enumerate(tuple(model.vars(_(px.NodeVar)))[0:-1]):
                    g[id(v)] = gs[i][id(v)] + gs[i + 1][id(v)]

                # g, (v,) = train_x(x)
                optim_x(g)

        with px.train(model):
            g, (v,) = train_w(x)
            optim_w(g)


@px.jit()
@px.bind(model)
def evaluate(x, y):
    with px.eval(model):
        y_hat, = predict(x, y)

    return (y_hat.argmax(-1) == y.argmax(-1)).mean()


def epoch(dl):
    x, y = next(iter(dl))
    y = one_hot(y, 10)

    for i in range(256):
        train_on_batch(x, y)

    return 0


def test(dl):
    accuracies = []
    for batch in dl:
        x, y = batch
        y = one_hot(y, 10)

        accuracies.append(evaluate(x, y))

    return np.mean(accuracies)


if __name__ == "__main__":
    t = timeit.timeit(lambda: epoch(train_dataloader), number=1)
    print("Compiling + Epoch 1 took", t, "seconds")

    # Time of an epoch (without jitting)
    t = timeit.timeit(lambda: epoch(train_dataloader), number=params["num_epochs"]) / params["num_epochs"]
    print("An Epoch takes on average", t, "seconds")

    print("Final Accuracy:", test(test_dataloader))

    del train_dataloader
    del test_dataloader
