# Core dependencies
import jax
import optax

# pcax
import pcax as px  # same as import pcax.pc as px
import pcax.nn as nn
import pcax.utils as pxu
from pcax.core import f  # _ is the filter object, more about it later!
from pcax.utils.data import TorchDataloader

import numpy as np
from torchvision.datasets import MNIST
import timeit
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Model(px.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 nm_layers: int,
                 act_fn=jax.nn.tanh) -> None:
        super().__init__()

        self.act_fn = act_fn

        self.layers = [nn.Linear(input_dim, hidden_dim)] + [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 2)
        ] + [nn.Linear(hidden_dim, output_dim)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.act_fn(layer(x))

        x = self.layers[-1](x)

        return x


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


@pxu.grad_and_values(px.f(px.LayerParam))
@pxu.vectorize(in_axis=(0, 0), out_axis=("sum", 0))
def train(x: jax.Array, t: jax.Array, *, model=None):
    y = model(x)
    # we also return y even if we will not use it, to show how it can be done. Note the change in out_axis.
    return (jax.numpy.square(y - t)).sum(), y  # it's MSE loss without the Mean, since in pc we use the sum


@pxu.jit()
def train_on_batch(x, y, *, model, optim):
    """
    Again, we don't need to initialize any node so we don't need to pass any argument to 'pxu.train'.
    If we need the predicted y_hat we can use the value returned by the 'train' function.
    """
    with pxu.train(model):
        g, (v, y_hat) = train(x, y, model=model)
        optim(g)


@pxu.jit()
def evaluate(x, y, *, model):
    with pxu.eval(model, x) as (y_hat,):
        return (y_hat.argmax(-1) == y.argmax(-1)).mean()


def epoch(train_fn, dl=None, train_data=None):
    if dl is not None:
        for batch in dl:
            x, y = batch
            y = one_hot(y, 10)

            train_fn(x, y)
    elif train_data is not None:
        x, y, i = train_data

        for i in range(i):
            train_fn(x, y)

    return 0


def test(test_fn, dl):
    accuracies = []
    for batch in dl:
        x, y = batch
        y = one_hot(y, 10)

        accuracies.append(test_fn(x, y))

    return np.mean(accuracies)


if __name__ == "__main__":
    params = {
        "batch_size": 128,
        "w_learning_rate": 1e-4,
        "num_epochs": 8,
        "num_layers": 8,
        "hidden_dim": 128,
        "input_dim": 28 * 28,
        "output_dim": 10,
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

    model = Model(28 * 28, params["hidden_dim"], 10, nm_layers=params["num_layers"])

    # dummy run to init the optimizer parameters
    with pxu.train(model):
        optim = pxu.Optim(
            optax.adam(params["w_learning_rate"]),
            model.parameters().filter(f(px.LayerParam)),
        )

    x, y = next(iter(train_dataloader))
    y = one_hot(y, 10)

    train_fn = train_on_batch.snapshot(model=model, optim=optim)
    test_fn = evaluate.snapshot(model=model)

    t = timeit.timeit(lambda: epoch(train_fn, train_data=(x, y, len(train_dataloader))), number=1)
    print("Compiling + Epoch 1 took", t, "seconds")

    # warmup
    epoch(train_fn, train_data=(x, y, len(train_dataloader)))

    # Time of an epoch (without jitting)
    t = timeit.timeit(
        lambda: epoch(train_fn, train_data=(x, y, len(train_dataloader))),
        number=params["num_epochs"]
    ) / params["num_epochs"]
    print("An Epoch takes on average", t, "seconds")

    print("Final Accuracy:", test(test_fn, test_dataloader))

    del train_dataloader
    del test_dataloader
