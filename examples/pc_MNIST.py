import jax
import optax
import numpy as np
from torchvision.datasets import MNIST
import timeit

import pcax as px
import pcax.nn as nn
from pcax.core import _
from pcax.utils.data import TorchDataloader

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

NM_LAYERS = 2


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
        self.pc_h = nn.ModuleList([px.Layer() for _ in range(nm_layers)])
        self.pc2 = px.Layer()

        self.pc2.x.frozen = True

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
    "batch_size": 256,
    "x_learning_rate": 0.01,
    "w_learning_rate": 1e-3,
    "num_epochs": 4,
    "hidden_dim": 128,
    "input_dim": 28 * 28,
    "output_dim": 10,
    "seed": 0,
    "T": (NM_LAYERS + 2),
}

train_dataset = MNIST(
    "/tmp/mnist/",
    transform=FlattenAndCast(),
    train=True,
)
train_dataloader = TorchDataloader(
    train_dataset,
    batch_size=params["batch_size"],
    num_workers=8,
    shuffle=True,
    persistent_workers=True,
    pin_memory=True,
)

test_dataset = MNIST(
    "/tmp/mnist/",
    transform=FlattenAndCast(),
    train=False,
)
test_dataloader = TorchDataloader(
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


@px.vectorize(_(px.NodeVar), out_axis=("sum",))
@px.bind(model)
def train_op(x):
    model(x)
    return model.energy


train_x = px.gradvalues(
    _(px.NodeVar)(frozen=False),
)(train_op)
train_w = px.gradvalues(
    _[px.TrainVar, -_(px.NodeVar)],
)(train_op)


# dummy run to init the optimizer parameters
with px.eval(model):
    predict(np.zeros((params["batch_size"], 28 * 28)), None)
    optim_x = px.Optim(
        optax.sgd(params["x_learning_rate"]), model.vars(_(px.NodeVar)(frozen=False))
    )
    optim_w = px.Optim(
        optax.adam(params["w_learning_rate"]),
        model.vars(_[px.TrainVar, -_(px.NodeVar)]),
    )


@px.jit()
@px.bind(model, optim_w=optim_w, optim_x=optim_x)
def train_on_batch(x, y):
    with px.eval(model):
        with px.train(model):
            predict(x, y)

        for i in range(params["T"]):
            with px.train(model):
                g, (v,) = train_x(x)
                optim_x(g)

        with px.train(model):
            g, (v,) = train_w(x)
            optim_w(g)


@px.jit()
@px.bind(model)
def evaluate(x, y):
    with px.eval(model):
        y_hat, = predict(x, None)

    return (y_hat.argmax(-1) == y.argmax(-1)).mean()


def epoch(dl):
    for x, y in dl:
        y = one_hot(y, 10)

        train_on_batch(x, y)

    return 0


def test(dl):
    accuracies = []
    for batch in dl:
        x, y = batch
        y = one_hot(y, 10)

        a = evaluate(x, y)
        accuracies.append(a)

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
