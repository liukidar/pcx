import jax
import optax
import numpy as np
from torchvision.datasets import MNIST
import timeit

import pcax as px
import pcax.nn as nn
import pcax.core as pxc
from pcax.core import _
from pcax.utils.data import TorchDataloader


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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

    def __call__(self, x, t=None):
        x = self.linear1(self.act_fn(x))

        for i in range(len(self.linear_h)):
            x = self.linear_h[i](self.act_fn(x))

        x = self.linear2(self.act_fn(x))

        return x


def one_hot(x, k):
    return np.array(x[:, None] == np.arange(k), dtype=np.float32)


class FlattenAndCast:
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=np.float32) / 255.0)


params = {
    "batch_size": 256,
    "w_learning_rate": 1e-3,
    "num_epochs": 4,
    "hidden_dim": 128,
    "input_dim": 28 * 28,
    "output_dim": 10,
    "seed": 0,
}

train_dataset = MNIST(
    "/tmp/mnist/",
    transform=FlattenAndCast(),
    train=True,
)
train_dataloader = TorchDataloader(
    train_dataset,
    batch_size=params["batch_size"],
    num_workers=4,
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
    persistent_workers=True,
    pin_memory=True,
)

model = Model(28 * 28, params["hidden_dim"], 10)


@px.vectorize(_(px.NodeVar), in_axis=(0, 0))
@px.bind(model)
def predict(x, t):
    return model(x, t)


@px.vectorize(_(px.NodeVar), in_axis=(0, 0), out_axis=("sum",))
@px.bind(model)
def train_op(x, y):
    y_hat = model(x)
    return ((y - y_hat)**2).sum()


train_W = px.gradvalues(
    _(pxc.TrainVar),
)(train_op)


# dummy run to init the optimizer parameters
with px.eval(model):
    predict(np.zeros((params["batch_size"], 28 * 28)), (None,) * params["batch_size"])
    optim_w = px.Optim(
        optax.adam(params["w_learning_rate"]),
        model.vars(_(pxc.TrainVar)),
    )


@pxc.Jit
@px.bind(model, optim_w=optim_w)
def train_on_batch(x, y):
    with px.train(model):
        g, (v,) = train_W(x, y)
        optim_w(g)


@pxc.Jit
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
