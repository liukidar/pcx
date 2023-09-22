# Core dependencies
import jax
import optax

# pcax
import pcax as px  # same as import pcax.pc as px
import pcax.nn as nn
import pcax.utils as pxu
import pcax.core as pxc
from pcax.core import f
from pcax.utils.data import TorchDataloader

import numpy as np
from torchvision.datasets import MNIST
import timeit


class Model(px.EnergyModule):
    def __init__(self, input_dim, hidden_dim, output_dim, nm_layers) -> None:
        super().__init__()

        self.act_fn = jax.nn.tanh

        """
        This is quite standard. We define the layers and links (one layer for each link).
        """
        input_layer = nn.Linear(input_dim, hidden_dim)
        self.fc_layers = [input_layer] \
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 2)] \
            + [nn.Linear(hidden_dim, output_dim)]

        self.pc_nodes = [px.Node() for _ in range(nm_layers)]

        """
        We normally use the x of the last layer as the target, therefore we don't want to update it.
        """
        self.pc_nodes[-1].x.frozen = True

    """
    Here things are a bit different. __call__ accepts an optional target t (used during training),
    which is used to set the x of the last layer.
    """
    @pxc.pure_fn
    def __call__(self, x, t=None):
        """
        !!! IMPORTANT !!!
        Each (pc) layer contains a cache the stores the important intermediate values computed in the forward pass.
        By default, these are the incoming activation (u), the node values (x) and the energy (e).
        You can access them by using the [] operator, e.g., self.pc["x"].
        """
        for n, l in zip(self.pc_nodes[:-1], self.fc_layers[:-1]):
            x = n(self.act_fn(l(x)))["x"]

        x = self.pc_nodes[-1](self.fc_layers[-1](x))["x"]

        if t is not None:
            self.pc_nodes[-1]["x"] = t

        """
        The output of the network is the activation received by the last layer (since its x is clamped to the label).
        """
        return self.pc_nodes[-1]["u"]


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


@pxu.grad_and_values(filter=f(px.NodeParam)(frozen=False))
@pxu.vectorize(filter=f(px.NodeParam, with_cache=True), in_axis=(0,), out_axis=("sum",))
def train_x(x, *, model):
    model(x)

    return model.energy()


@pxu.grad_and_values(filter=f(px.LayerParam))
@pxu.vectorize(filter=f(px.NodeParam, with_cache=True), in_axis=(0,), out_axis=("sum",))
def train_w(x, *, model):
    model(x)

    return model.energy()


@pxu.jit()
def train_on_batch(x, y, *, model, optim_w, optim_x):
    def x_step(x, *, model, optim_x):
        with pxu.step(model):
            g, (v,) = train_x(x, model=model)
            optim_x(g)

    # We are working on the input x, so we initialise the internal nodes with it (this also initialises the cache).
    with pxu.train(model, x, y) as (y_hat,):
        model.clear_cache()
        pxu.flow.scan(x_step, length=params["T"])(x, model=model, optim_x=optim_x)

        # for i in range(params["T"]):
        #     # Each forward pass caches the intermediate values (such as activations and energies), so we can use them
        #     # to compute the gradients.
        #     # px.init_cache takes care of managing the cache.
        #     # !!! IMPORTANT: px.init_cache must always be used inside a px.init_nodes context !!!
        #     with pxu.step(model):
        #         g, (v,) = train_x(x, model=model)
        #         optim_x(g)

        # !!! IMPORTANT: px.init_cache must always be used inside a px.init_nodes context !!!
        with pxu.step(model):
            g, (v,) = train_w(x, model=model)
            optim_w(g)


@pxu.jit()
def evaluate(x, y, *, model):
    # As in train_on_batch, we initialise the internal nodes with the input x. By doing so we also get the model's
    # output y_hat.
    with pxu.eval(model, x, y) as (y_hat,):
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


def test(evaluate_fn, dl):
    accuracies = []
    for batch in dl:
        x, y = batch
        y = one_hot(y, 10)

        accuracies.append(evaluate_fn(x, y))

    return np.mean(accuracies)


if __name__ == "__main__":
    params = {
        "batch_size": 128,
        "x_learning_rate": 0.05,
        "w_learning_rate": 5e-4,
        "num_epochs": 8,
        "num_layers": 2,
        "hidden_dim": 256,
        "input_dim": 28 * 28,
        "output_dim": 10,
        "T": 4,
    }
    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=params["batch_size"],
        num_workers=8,
        shuffle=True,
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
    # load_params(model.parameters().filter(_(px.LinkVar)), "mnist_params.npz")

    # dummy run to init the optimizer parameters
    with pxu.train(model, np.zeros((params["batch_size"], 28 * 28)), None):
        optim_x = pxu.Optim(
            optax.sgd(params["x_learning_rate"]), model.parameters().filter(f(px.NodeParam)(frozen=False)),
        )
        optim_w = pxu.Optim(
            optax.adam(params["w_learning_rate"]),
            model.parameters().filter(f(px.LayerParam)),
        )

    x, y = next(iter(train_dataloader))
    y = one_hot(y, 10)

    train_fn = train_on_batch.snapshot(model=model, optim_w=optim_w, optim_x=optim_x)
    test_fn = evaluate.snapshot(model=model)

    t = timeit.timeit(lambda: epoch(train_fn, dl=train_dataloader), number=1)
    print("Compiling + Epoch 1 took", t, "seconds")

    # warmup
    epoch(train_fn, dl=train_dataloader)

    # Time of an epoch (without having to jit)
    t = timeit.timeit(
        lambda: epoch(train_fn, dl=train_dataloader),
        number=params["num_epochs"]
    ) / params["num_epochs"]
    print("An Epoch takes on average", t, "seconds")

    print("Final Accuracy:", test(test_fn, test_dataloader))

    del train_dataloader
    del test_dataloader
