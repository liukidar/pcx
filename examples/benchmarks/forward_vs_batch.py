from typing import Callable, Optional
from tqdm import tqdm
import pandas as pd
import torch
from torchvision.datasets import FashionMNIST

# Core dependencies
import numpy as np
import jax
import jax.numpy as jnp
import optax

# pcax
import pcax as px
import pcax.core as pxc
import pcax.utils as pxu
import pcax.nn as nn

# Environment variables
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    os.mkdir("forward_vs_batch")
except FileExistsError:
    pass


def ce_energy(node: px.Node, rkg: pxc.RandomKeyGenerator):
    u = jax.nn.softmax(node["u"])
    return (node["x"] * jnp.log(node["x"] / (u + 1e-8) + 1e-8)).sum(axis=-1)


class Model(px.EnergyModule):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 nm_layers: int,
                 act_fn: Callable[[jax.Array], jax.Array],
                 init_fn: Optional[Callable[[px.Node, pxc.RandomKeyGenerator], None]] = None
                 ) -> None:
        super().__init__()

        self.nm_classes = output_dim
        self.act_fn = act_fn
        self.layers = [nn.Linear(input_dim, hidden_dim)] + [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 1)
        ] + [nn.Linear(hidden_dim, output_dim)]

        self.nodes = [px.Node(init_fn=init_fn) for _ in range(nm_layers)] + [px.Node(energy_fn=ce_energy)]

        self.nodes[-1].x.frozen = True

    def __call__(self,
                 x: jax.Array,
                 t: Optional[jax.Array] = None,
                 forward_to: Optional[int] = None):
        if forward_to is None:
            forward_to = len(self.nodes)

        for i, (node, layer) in enumerate(zip(self.nodes[:forward_to], self.layers[:forward_to])):
            act_fn = self.act_fn if i != len(self.nodes) - 1 else lambda x: x
            x = node(act_fn(layer(x)))["x"]

        if t is not None:
            self.nodes[-1]["x"] = t

        return self.nodes[-1]["u"]


@pxu.vectorize(px.f(px.NodeParam, with_cache=True), in_axis=(0, 0), out_axis=("sum",))
def loss(x: jax.Array, t: Optional[jax.Array] = None, *, model):
    model(x)
    return model.energy()


train_x = pxu.grad_and_values(
    px.f(px.NodeParam)(frozen=False),
)(loss)


train_w = pxu.grad_and_values(
    px.f(px.LayerParam)
)(loss)


train = pxu.grad_and_values(
    px.f(px.LayerParam) | px.f(px.NodeParam)(frozen=False)
)(loss)


@pxu.jit()
def train_batch(x, y, *, T, model, optim_w, optim_x, mode):
    y = jax.nn.one_hot(y, 10)

    with pxu.train(model, x, y) as (y_hat,):
        if mode == "pc":
            for i in range(T):
                with pxu.step(model):
                    g, (v,) = train_x(x, y, model=model)
                    optim_x(g)

            with pxu.step(model):
                g, (v,) = train_w(x, y, model=model)
                optim_w(g)

        elif mode == "ppc":
            for i in range(T):
                with pxu.step(model):
                    g, (v,) = train(x, y, model=model)
                    optim_x(g)
                    optim_w(g)


@pxu.jit()
def eval_batch(x, y, *, model):
    y = jax.nn.one_hot(y, 10)
    with pxu.eval(model, x, y) as (y_hat,):
        return (y_hat.argmax(-1) == y.argmax(-1)).mean()


def epoch(dl, train_fn):
    for i, batch in enumerate(dl):
        x, y = batch

        train_fn(x, y)


@pxu.jit()
def train_batch_avg_init(x, y, nodes_avg_by_class, *, T, model, optim_w, optim_x, mode, n):
    if nodes_avg_by_class is None:
        n = len(model.nodes)

    with pxu.train(model, x, forward_to=n):
        # Initialize nodes
        if nodes_avg_by_class is not None:
            for node, avg_by_class in zip(model.nodes[n:-1], nodes_avg_by_class):
                node["x"] = jnp.repeat(avg_by_class, node["x"].shape[0] // model.nm_classes, axis=0)

        # Convert y to one_hot
        y = jax.nn.one_hot(y, 10)

        # Set last node to y
        model.nodes[-1]["x"] = y

        if mode == "pc":
            for i in range(T):
                with pxu.step(model):
                    g, (v,) = train_x(x, y, model=model)
                    optim_x(g)

            with pxu.step(model):
                g, (v,) = train_w(x, y, model=model)
                optim_w(g)
        elif mode == "ppc":
            for i in range(T):
                with pxu.step(model):
                    g, (v,) = train(x, y, model=model)
                    optim_x(g)
                    optim_w(g)

    nodes_avg_by_class = [
        jnp.mean(jnp.reshape(node["x"], (model.nm_classes, -1, *node["x"].shape[1:])), axis=1)
        for node in model.nodes[n:-1]
    ]

    return nodes_avg_by_class


def epoch_avg_init(dl, train_fn):
    nodes_avg_by_class = None
    for i, batch in enumerate(dl):
        x, y = batch
        nodes_avg_by_class = train_fn(x, y, nodes_avg_by_class)


def test(dl, eval_fn):
    accuracies = []
    for batch in dl:
        x, y = batch

        accuracies.append(eval_fn(x, y))

    return np.mean(accuracies)


def warmup(dl, train_fn, warmup_steps: int):
    for i, batch in enumerate(dl):
        x, y = batch

        train_fn(x, y)

        if i == warmup_steps:
            break


# Define parameters
params = {
    "batch_size": 300,
    "w_learning_rate": 2e-4,
    "num_epochs": 16,
    "num_layers": 2,
    "hidden_dim": 512,
    "input_dim": 28 * 28,
    "output_dim": 10,
    "p": 0.25,
    "mode": "pc"
}


# Used to convert the square 0-255 images to 0-1 float vectors.
class FlattenAndCast:
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=np.float32) / 255.0)


# Create dataloaders
train_dataset = FashionMNIST(
    "/tmp/mnist/",
    transform=FlattenAndCast(),
    download=True,
    train=True,
)
train_dataset = torch.utils.data.Subset(train_dataset, indices=np.arange(0, int(len(train_dataset) * params["p"])))
train_dataloader = pxu.data.TorchDataloader(
    train_dataset,
    batch_sampler=pxu.data.BatchAlignedSampler(train_dataset, params["batch_size"]),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

test_dataset = FashionMNIST(
    "/tmp/mnist/",
    transform=FlattenAndCast(),
    download=True,
    train=False,
)
test_dataloader = pxu.data.TorchDataloader(
    test_dataset,
    batch_size=params["batch_size"],
    shuffle=False,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

TRIALS_PER_SEED = 3

if __name__ == "__main__":
    accuracies = {}

    Ts = [2, 3, 4, 5, 6, 7, 8, 9]
    XLRs = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]

    for i, xlr in enumerate(XLRs):
        params["x_learning_rate"] = xlr

        # Create model
        model = Model(28 * 28, params["hidden_dim"], 10, params["num_layers"], jax.nn.gelu)

        # Create optimizers
        with pxu.train(model, jnp.zeros((params["batch_size"], 28 * 28)), None):
            optim_x = pxu.Optim(
                optax.sgd(params["x_learning_rate"]),
                model.parameters().filter(px.f(px.NodeParam)(frozen=False))
            )
            optim_w = pxu.Optim(
                optax.adamw(params["w_learning_rate"]),
                model.parameters().filter(px.f(px.LayerParam)),
            )

        # Create snapshots
        eval_fn = eval_batch.snapshot(model=model)

        iter_bar = tqdm(Ts)
        for T in iter_bar:
            params["T"] = T
            iter_bar.set_description(f"{i + 1}/{len(XLRs)}")

            train_fn = train_batch.snapshot(T=params["T"], model=model, optim_x=optim_x, optim_w=optim_w,
                                            mode=params["mode"])
            train_avg_init_fn = train_batch_avg_init.snapshot(T=params["T"], model=model, optim_x=optim_x,
                                                              optim_w=optim_w, n=params["num_layers"] // 2,
                                                              mode=params["mode"])

            for mode in ["normal", "avg_init"]:
                accuracy_over_seeds = []
                for seed in range(TRIALS_PER_SEED):
                    # Create new model
                    px.move(
                        Model(28 * 28, params["hidden_dim"], 10, params["num_layers"], jax.nn.gelu)
                        .parameters().filter(px.f(px.LayerParam)),
                        model.parameters().filter(px.f(px.LayerParam))
                    )

                    # Create new w optimizer (x has no state)
                    px.move(
                        pxu.Optim(
                            optax.adamw(params["w_learning_rate"]),
                            model.parameters().filter(px.f(px.LayerParam))
                        ).parameters(),
                        optim_w.parameters()
                    )

                    # Warmup:
                    warmup(train_dataloader, train_fn, 8)

                    # Train:
                    accuracies_over_epoch = []
                    if mode == "normal":
                        for e in range(params["num_epochs"]):
                            epoch(train_dataloader, train_fn)
                            accuracies_over_epoch.append(test(test_dataloader, eval_fn))
                    elif mode == "avg_init":
                        for e in range(params["num_epochs"]):
                            epoch_avg_init(train_dataloader, train_avg_init_fn)
                            accuracies_over_epoch.append(test(test_dataloader, eval_fn))

                    accuracy_over_seeds.append(accuracies_over_epoch)

                accuracy_over_seeds = np.mean(accuracy_over_seeds, axis=0)
                for e in range(params["num_epochs"]):
                    accuracies[(T, e, xlr, mode)] = accuracy_over_seeds[e]

    with open(
        f"forward_vs_batch/{params['hidden_dim']}_{params['num_layers']}_{params['p']}_adamw_{params['mode']}.csv",
        "w"
    ) as f:
        df = pd.DataFrame(list(accuracies.items()), columns=["(T, E, XLR, M)", "accuracy"])
        df[["T", "E", "XLR", "M"]] = pd.DataFrame(df["(T, E, XLR, M)"].tolist(), index=df.index)
        df.drop(columns=["(T, E, XLR, M)"], inplace=True)
        df.to_csv(f)
