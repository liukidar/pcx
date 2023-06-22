# Core dependencies
import jax
import optax

# pcax
import pcax as px  # same as import pcax.pc as px
import pcax.nn as nn
from pcax.core import f  # _ is the filter object, more about it later!
from pcax.utils.data import TorchDataloader

import numpy as np
from torchvision.datasets import MNIST
import timeit
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

NM_LAYERS = 2


class Model(px.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nm_layers=NM_LAYERS) -> None:
        super().__init__()

        self.act_fn = jax.nn.tanh

        """
        This is quite standard. We define the layers and links (one layer for each link).
        """
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear_h = [nn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers)]
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.pc1 = px.Layer()
        self.pc_h = [px.Layer() for _ in range(nm_layers)]
        self.pc2 = px.Layer()

        """
        We normally use the x of the last layer as the target, therefore we don't want to update it.
        """
        self.pc2.x.frozen = True

    """
    Here things are a bit different. __call__ accepts an optional target t (used during training),
    which is used to set the x of the last layer.
    """
    def __call__(self, x, t=None):
        """
        !!! IMPORTANT !!!
        Each (pc) layer contains a cache the stores the important intermediate values computed in the forward pass.
        By default, these are the incoming activation (u), the node values (x) and the energy (e).
        You can access them by using the [] operator, e.g., self.pc["x"].
        """
        x = self.pc1(self.act_fn(self.linear1(x)))["x"]

        for i in range(len(self.linear_h)):
            x = self.pc_h[i](self.act_fn(self.linear_h[i](x)))["x"]

        x = self.pc2(self.linear2(x))["x"]

        if t is not None:
            self.pc2["x"] = t

        """
        The output of the network is the activation received by the last layer (since its x is clamped to the label).
        """
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
    "~/tmp/mnist/",
    transform=FlattenAndCast(),
    train=True,
    download=True
)
train_dataloader = TorchDataloader(
    train_dataset,
    batch_size=params["batch_size"],
    num_workers=16,
    shuffle=True,
    persistent_workers=True,
    pin_memory=True,
)

test_dataset = MNIST(
    "~/tmp/mnist/",
    transform=FlattenAndCast(),
    train=False,
    download=True
)
test_dataloader = TorchDataloader(
    test_dataset,
    batch_size=params["batch_size"],
    num_workers=16,
    shuffle=False,
    persistent_workers=False,
    pin_memory=True,
)


@px.gradvalues(filter=f(px.NodeVar)(frozen=False))
@px.vectorize(filter=f(px.NodeVar, with_cache=True), in_axis=(0,), out_axis=("sum",))
def train_x(x, *, model):
    model(x)
    return model.energy()


@px.gradvalues(filter=f(px.LinkVar))
@px.vectorize(filter=f(px.NodeVar, with_cache=True), in_axis=(0,), out_axis=("sum",))
def train_w(x, *, model, dummy_static_arg):
    if dummy_static_arg is True:
        model(x)
    else:
        model(x)

    return model.energy()


@px.jit()
def train_on_batch(x, y, *, model, optim_w, optim_x):
    # We are working on the input x, so we initialise the internal nodes with it (this also initialises the cache).
    with px.init_nodes(model, x, y) as (y_hat,):
        for i in range(params["T"]):
            # Each forward pass caches the intermediate values (such as activations and energies), so we can use them
            # to compute the gradients.
            # px.init_cache takes care of managing the cache.
            # !!! IMPORTANT: px.init_cache must always be used inside a px.init_nodes context !!!
            with px.init_cache(model):
                g, (v,) = train_x(x, model=model)
                optim_x(g)

        # !!! IMPORTANT: px.init_cache must always be used inside a px.init_nodes context !!!
        with px.init_cache(model):
            g, (v,) = train_w(x, model=model, dummy_static_arg=True)
            optim_w(g)


@px.jit()
def evaluate(x, y, *, model):
    # As in train_on_batch, we initialise the internal nodes with the input x. By doing so we also get the model's
    # output y_hat.
    with px.init_nodes(model, x, y) as (y_hat,):
        return (y_hat.argmax(-1) == y.argmax(-1)).mean()


def epoch(train_fn, dl):
    for batch in dl:
        x, y = batch
        y = one_hot(y, 10)
        # Static arguments in the first brackets, dynamic arguments in the second.
        train_fn(x, y)

    return 0


def test(evaluate_fn, dl):
    accuracies = []
    for batch in dl:
        x, y = batch
        y = one_hot(y, 10)

        # Static arguments in the first brackets, dynamic arguments in the second.
        accuracies.append(evaluate_fn(x, y))

    return np.mean(accuracies)


model = Model(28 * 28, params["hidden_dim"], 10)
# load_params(model.parameters().filter(_(px.LinkVar)), "mnist_params.npz")

# dummy run to init the optimizer parameters
with px.init_nodes(model, np.zeros((params["batch_size"], 28 * 28)), None):
    optim_x = px.Optim(
        optax.sgd(params["x_learning_rate"]), model.parameters().filter(f(px.NodeVar)(frozen=False)),
    )
    optim_w = px.Optim(
        optax.adam(params["w_learning_rate"]),
        model.parameters().filter(f(px.LinkVar)),  # same as _(...) - _(...)
    )

if __name__ == "__main__":
    train_fn = train_on_batch.snapshot(model=model, optim_w=optim_w, optim_x=optim_x)
    test_fn = evaluate.snapshot(model=model)

    t = timeit.timeit(lambda: epoch(train_fn, train_dataloader), number=1)
    print("Compiling + Epoch 1 took", t, "seconds")

    # Time of an epoch (without jitting)
    t = timeit.timeit(lambda: epoch(train_fn, train_dataloader), number=params["num_epochs"]) / params["num_epochs"]
    print("An Epoch takes on average", t, "seconds")

    print("Final Accuracy:", test(test_fn, test_dataloader))

    del train_dataloader
    del test_dataloader
