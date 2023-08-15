# Core dependencies
import jax
import optax

# pcax
import pcax as px  # same as import pcax.pc as px
import pcax.nn as nn
from pcax.core import f  # _ is the filter object, more about it later!
from pcax.utils.data import TorchDataloader
import pcax.experimental

import numpy as np
from torchvision.datasets import MNIST
import timeit
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class Model(px.EnergyModule):
    def __init__(self, input_dim, hidden_dim, output_dim, nm_layers) -> None:
        super().__init__()

        self.act_fn = jax.nn.tanh

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.pexec_mlb = pcax.experimental.pexec_MultiLinearBlock(hidden_dim, nm_layers, self.act_fn)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.hidden_node = px.Node()
        self.output_node = px.Node()

        self.output_node.x.frozen = True

    def __call__(self, x, t=None):
        x = self.act_fn(self.input_layer(x))
        x = self.hidden_node(self.pexec_mlb(x))["x"]
        x = self.output_node(self.output_layer(x))

        if t is not None:
            self.output_node["x"] = t

        return self.output_node["u"]


def one_hot(x, k):
    return np.array(x[:, None] == np.arange(k), dtype=np.float32)


class FlattenAndCast:
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=np.float32) / 255.0)


train_dataset = MNIST(
    "~/tmp/mnist/",
    transform=FlattenAndCast(),
    train=True,
    download=True
)

test_dataset = MNIST(
    "~/tmp/mnist/",
    transform=FlattenAndCast(),
    train=False,
    download=True
)


@px.grad_and_values(filter=f(px.NodeParam)(frozen=False))
@px.vectorize(filter=f(px.NodeParam, with_cache=True), in_axis=(0,), out_axis=("sum",))
def train_x(x, *, model):
    model(x)
    return model.energy()


@px.grad_and_values(filter=f(px.LayerParam))
@px.vectorize(filter=f(px.NodeParam, with_cache=True), in_axis=(0,), out_axis=("sum",))
def train_w(x, *, model):
    model(x)

    return model.energy()


@px.jit()
def train_on_batch(x, y, *, model, optim_w, optim_x):
    with px.train(model, x, y) as (y_hat,):
        for i in range(params["T"]):
            with px.step(model):
                g, (v,) = train_x(x, model=model)
                optim_x(g)

        with px.step(model):
            g, (v,) = train_w(x, model=model)
            optim_w(g)


@px.jit()
def evaluate(x, y, *, model):
    with px.eval(model, x, y) as (y_hat,):
        return (y_hat.argmax(-1) == y.argmax(-1)).mean()


def epoch(train_fn, x, y, i):
    for i in range(i):
        train_fn(x, y)

    return 0


def test(evaluate_fn, dl):
    accuracies = []
    for batch in dl:
        x, y = batch
        y = one_hot(y, 10)

        accuracies.append(evaluate_fn(x, y))

    return np.mean(accuracies)


if __name__ == "__main__":
    params = {
        "batch_size": 256,
        "x_learning_rate": 0.05,
        "w_learning_rate": 1e-4,
        "num_epochs": 8,
        "num_layers": 10,
        "hidden_dim": 128,
        "input_dim": 28 * 28,
        "output_dim": 10,
        "T": 8,
    }
    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=params["batch_size"],
        num_workers=8,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )
    test_dataloader = TorchDataloader(
        test_dataset,
        batch_size=params["batch_size"],
        num_workers=8,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )

    model = Model(784, params["hidden_dim"], 10, nm_layers=params["num_layers"])

    with px.eval(model, np.zeros((params["batch_size"], 28 * 28)), None):
        optim_x = px.Optim(
            optax.chain(
                optax.sgd(params["x_learning_rate"])
            ), model.parameters().filter(f(px.NodeParam)(frozen=False)),
        )
        optim_w = px.Optim(
            optax.adam(params["w_learning_rate"]),
            model.parameters().filter(f(px.LayerParam)),
        )

    x, y = next(iter(train_dataloader))
    y = one_hot(y, 10)

    train_fn = train_on_batch.snapshot(model=model, optim_w=optim_w, optim_x=optim_x)
    test_fn = evaluate.snapshot(model=model)

    t = timeit.timeit(lambda: epoch(train_fn, x, y, len(train_dataloader)), number=1)
    print("Compiling + Epoch 1 took", t, "seconds")

    # warmup
    epoch(train_fn, x, y, len(train_dataloader))
    # Time of an epoch (without jitting)
    t = timeit.timeit(
        lambda: epoch(train_fn, x, y, len(train_dataloader)),
        number=params["num_epochs"]
    ) / params["num_epochs"]
    print("An Epoch takes on average", t, "seconds")

    print("Final Accuracy:", test(test_fn, test_dataloader))

    del train_dataloader
    del test_dataloader
