import os
import gc
from typing import Callable
from argparse import ArgumentParser

# from functools import partial
import pathlib

# from itertools import islice
import pprint

import jax
import jax.numpy as jnp
import optax
import pcax as px
import pcax.utils as pxu
import pcax.nn as nn
import pcax.core as pxc
import torch
from ray.air import session
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from params import Params

# Environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class PCGenerator(px.EnergyModule):
    def __init__(
        self,
        internal_dim: int,
        output_dim: int,
        num_layers: int,
        act_fn: Callable[[jax.Array], jax.Array],
        internal_init_fn: Callable[[px.Node, pxc.RandomKeyGenerator, int], None],
    ) -> None:
        super().__init__()

        self.internal_dim = internal_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.act_fn = act_fn

        dim_step = round((output_dim - internal_dim) / num_layers)

        self.fc_layers = [
            nn.Linear(
                internal_dim + dim_step * i, internal_dim + dim_step * (i + 1) if i + 1 != num_layers else output_dim
            )
            for i in range(num_layers)
        ]
        self.pc_nodes = [px.Node(init_fn=lambda node, rkg: internal_init_fn(node, rkg, internal_dim))] + [
            px.Node() for _ in range(num_layers)
        ]

        self.pc_nodes[-1].x.frozen = True

    def __call__(self, example: jax.Array = None, internal_state: jax.Array = None) -> jax.Array:
        if internal_state is None:
            internal_state = self.internal_state  # this might be None as well.
        # Call the internal layer so it can initialize the internal state by calling init_fn.
        x = self.pc_nodes[0](internal_state)["x"]

        for i in range(self.num_layers):
            act_fn = self.act_fn if i < self.num_layers - 1 else lambda x: x
            x = self.pc_nodes[i + 1](act_fn(self.fc_layers[i](x)))["x"]

        if example is not None:
            self.pc_nodes[-1]["x"] = example

        return self.prediction

    @property
    def prediction(self):
        # Return the output ("u" is equal to "x" if the target is not fixed,
        # while it is the actual output of the model if the target is fixed)
        return self.pc_nodes[-1]["u"]

    @property
    def internal_state(self):
        return self.pc_nodes[0]["x"]


@pxu.vectorize(px.f(px.NodeParam, with_cache=True), in_axis=(0,), out_axis=("sum",))
def loss(example: jax.Array, *, model) -> jax.Array:
    model(example=example)
    return model.energy()


@pxu.vectorize(px.f(px.NodeParam, with_cache=True), in_axis=(0,))
def predict(example: jax.Array, *, model) -> jax.Array:
    return model(example=example)


# class ReentryIsliceIterator:
#     def __init__(self, iterable, limit):
#         self.iterable = iterable
#         self.limit = limit

#     def __iter__(self):
#         return islice(self.iterable, self.limit).__iter__()


def get_data_loaders(params: Params) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(root=params.data_dir, train=True, download=True, transform=transform)  # type: ignore
    test_dataset = datasets.MNIST(root=params.data_dir, train=False, download=True, transform=transform)  # type: ignore

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        # drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        # drop_last=True,
    )

    # train_loader = ReentryIsliceIterator(train_loader, 1)
    # test_loader = ReentryIsliceIterator(test_loader, 1)

    return train_loader, test_loader


# def stats(gradients):
#     if not gradients:
#         return {}
#     return {
#         "has_nan": any([bool(jnp.isnan(x).any()) for x in gradients.values()]),
#         "has_inf": any([bool(jnp.isinf(x).any()) for x in gradients.values()]),
#         "max": max([float(x.max()) for x in gradients.values()]),
#         "min": min([float(x.min()) for x in gradients.values()]),
#     }


@pxu.jit()
def train_on_batch(examples: jax.Array, *, model: PCGenerator, optim_x, optim_w, loss, T):
    grad_and_values = pxu.grad_and_values(
        px.f(px.NodeParam)(frozen=False) | px.f(px.LayerParam),
    )(loss)

    # x_param_ids = {id(x) for x in model.parameters().filter(px.f(px.NodeParam)(frozen=False)).values()}
    # w_param_ids = {id(x) for x in model.parameters().filter(px.f(px.LayerParam)).values()}

    with pxu.train(model, examples):
        for i in range(T):
            with pxu.step(model):
                g, _ = grad_and_values(examples, model=model)
                # g_x = {k: v for k, v in g.items() if k in x_param_ids}
                # g_w = {k: v for k, v in g.items() if k in w_param_ids}
                # g_x_stats = stats(g_x)
                # g_w_stats = stats(g_w)
                # assert len(g) == len(g_x) + len(g_w)
                # assert not g_x_stats["has_nan"]
                # assert not g_x_stats["has_inf"]
                # assert not g_w_stats["has_nan"]
                # assert not g_w_stats["has_inf"]
                optim_x(g)
                optim_w(g)
    predictions = predict(examples, model=model)[0]
    mse = jnp.mean((predictions - examples) ** 2)
    return mse


@pxu.jit()
def test_on_batch(examples, *, model: PCGenerator, optim_x, loss, T):
    grad_and_values = pxu.grad_and_values(
        px.f(px.NodeParam)(frozen=False),
    )(loss)
    predictions = None
    with pxu.eval(model, examples):
        for i in range(T):
            with pxu.step(model):
                g, _ = grad_and_values(examples, model=model)

                optim_x(g)
    predictions = predict(examples, model=model)[0]
    mse = jnp.mean((predictions - examples) ** 2)
    return mse


@pxu.jit()
def get_internal_states_on_batch(examples, *, model: PCGenerator, optim_x, loss, T):
    grad_and_values = pxu.grad_and_values(
        px.f(px.NodeParam)(frozen=False),
    )(loss)
    with pxu.eval(model, examples):
        for i in range(T):
            with pxu.step(model):
                g, _ = grad_and_values(examples, model=model)

                optim_x(g)
    internal_states = model.internal_state
    return internal_states


def internal_init(node: px.Node, rkg: pxc.RandomKeyGenerator, internal_dim: int) -> None:
    value = jax.random.normal(rkg(), (internal_dim,))
    node["u"] = value
    node["x"] = value


def train_model(params: Params) -> None:
    model = PCGenerator(
        params.internal_dim, params.output_dim, params.num_layers, act_fn=jax.nn.tanh, internal_init_fn=internal_init
    )

    with pxu.train(model, jax.numpy.zeros((params.batch_size, params.output_dim))):
        optim_x = pxu.Optim(
            optax.chain(optax.add_decayed_weights(weight_decay=params.optim_x_l2), optax.sgd(params.optim_x_lr)),
            model.parameters().filter(px.f(px.NodeParam)(frozen=False)),
            allow_none_grads=True,
        )
        optim_w = pxu.Optim(
            optax.chain(
                optax.add_decayed_weights(weight_decay=params.optim_w_l2),
                optax.sgd(
                    params.optim_w_lr / params.batch_size,
                    momentum=params.optim_w_momentum,
                    nesterov=params.optim_w_nesterov,
                ),
            ),
            model.parameters().filter(px.f(px.LayerParam)),
        )

    train_batch_fn = train_on_batch.snapshot(
        model=model,
        optim_x=optim_x,
        optim_w=optim_w,
        loss=loss,
        T=params.T,
    )

    test_batch_fn = test_on_batch.snapshot(
        model=model,
        optim_x=optim_x,
        loss=loss,
        T=params.T,
    )

    train_loader, test_loader = get_data_loaders(params)

    train_mses = []
    test_mses = []

    with tqdm(range(params.epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Train Epoch {epoch + 1}")

            epoch_train_mses = []
            with tqdm(train_loader, unit="batch") as tbatch:
                for examples, _ in tbatch:
                    tbatch.set_description(f"Train Batch {tbatch.n + 1}")

                    examples = jax.numpy.array(examples.numpy()).reshape(-1, 784)
                    metric = train_batch_fn(examples)
                    epoch_train_mses.append(metric)
                    tbatch.set_postfix(mse=metric)
            epoch_train_mse = np.mean(epoch_train_mses)
            train_mses.append(epoch_train_mse)

            epoch_test_mses = []
            with tqdm(test_loader, unit="batch") as tbatch:
                for examples, _ in tbatch:
                    tbatch.set_description(f"Test Batch {tbatch.n + 1}")

                    examples = jax.numpy.array(examples.numpy()).reshape(-1, 784)
                    metric = test_batch_fn(examples)
                    epoch_test_mses.append(metric)
                    tbatch.set_postfix(mse=metric)
            epoch_test_mse = np.mean(epoch_test_mses)
            test_mses.append(epoch_test_mse)

            tepoch.set_postfix(train_mse=epoch_train_mse, test_mse=epoch_test_mse)

    assert len(train_mses) == len(test_mses)
    session.report(
        {
            "test_mse": test_mses[-1],
            "train_mse": train_mses[-1],
            "epochs": len(test_mses),
        }
    )

    result_dir = pathlib.Path(params.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    sns.lineplot(x=range(len(train_mses)), y=train_mses, ax=ax, label="Train MSE avg")
    sns.lineplot(x=range(len(test_mses)), y=test_mses, ax=ax, label="Test MSE avg")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE avg per epoch")
    ax.set_title("MSE avg per epoch")
    ax.legend()
    fig.savefig(result_dir / "pc_decoder_mnist.png")

    internal_states_data = {
        "x": [],
        "y": [],
        "label": [],
    }

    get_internal_states_fn = get_internal_states_on_batch.snapshot(
        model=model,
        optim_x=optim_x,
        loss=loss,
        T=params.T,
    )
    with tqdm(test_loader, unit="batch") as tbatch:
        for examples, labels in tbatch:
            tbatch.set_description(f"Test Batch {tbatch.n + 1}")

            examples = jax.numpy.array(examples.numpy()).reshape(-1, 784)
            states = get_internal_states_fn(examples)
            assert states.shape == (params.batch_size, params.internal_dim)
            assert labels.shape == (params.batch_size,)
            internal_states_data["x"].extend(states[:, 0].tolist())
            internal_states_data["y"].extend(states[:, 1].tolist())
            internal_states_data["label"].extend(labels.tolist())

    internal_states_df = pd.DataFrame(internal_states_data)

    plt.clf()
    sns.scatterplot(data=internal_states_df, x="x", y="y", hue="label")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Internal representations of classes")
    plt.legend()
    plt.savefig(result_dir / "pc_decoder_mnist_internal_states.png")


class Trainable:
    def __init__(self, params: Params) -> None:
        self.params = params

    def __call__(self, config: dict):
        gc.collect()
        params = self.params.update(config, inplace=False, validate=True)
        # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.utils.wait_for_gpu.html#ray.tune.utils.wait_for_gpu
        if params.hypertunning_gpu_memory_fraction_per_trial > 0:
            tune.utils.wait_for_gpu(
                target_util=1.0 - params.hypertunning_gpu_memory_fraction_per_trial,
                retry=50,
                delay_s=12,
            )
        train_model(params)
        gc.collect()


def main():
    parser = ArgumentParser()
    Params.add_arguments(parser)
    args = parser.parse_args()
    params = Params.from_arguments(args)
    pp = pprint.PrettyPrinter(indent=4)

    # Download datasets
    get_data_loaders(params)

    if params.hypertunning_gpu_memory_fraction_per_trial < 0 or params.hypertunning_gpu_memory_fraction_per_trial > 1:
        raise ValueError(f"--hypertunning-gpu-memory-fraction-per-trial must be in [0, 1]")

    if params.do_hypertunning:
        trainable = Trainable(params)
        trainable = tune.with_resources(
            trainable,
            {
                "cpu": params.hypertunning_cpu_per_trial,
                "gpu": params.hypertunning_gpu_memory_fraction_per_trial,
            },
        )
        param_space = params.ray_tune_param_space()
        search_algo = OptunaSearch(
            # points_to_evaluate=points_to_evaluate,
        )
        if params.hypertunning_max_concurrency is not None:
            search_algo = ConcurrencyLimiter(search_algo, max_concurrent=params.hypertunning_max_concurrency)
        scheduler = None
        if params.hypertunning_use_early_stop_scheduler:
            scheduler = ASHAScheduler()
        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                num_samples=params.hypertunning_num_trials,
                metric="test_mse",
                mode="min",
                search_alg=search_algo,
                scheduler=scheduler,
                reuse_actors=False,
            ),
            run_config=air.RunConfig(
                name=params.experiment_name,
                local_dir=params.result_dir,
                failure_config=air.FailureConfig(max_failures=3),
            ),
        )
        results = tuner.fit()
        print("--- Hypertunning done! ---")
        pp.pprint(results)
        print("--- Best trial ---")
        best_result = results.get_best_result()
        pp.pprint(best_result.config)
        pp.pprint(best_result.metrics)
        pp.pprint(best_result.log_dir)

        # FIXME: use the best reported score to compare results!
        # Note: ray.tune interprets the last reported result of each trial as the "best" one:
        # https://github.com/ray-project/ray_lightning/issues/81
        results.get_dataframe().to_csv(os.path.join(params.result_dir, "hypertunning_results.csv"), index=False)
    else:
        train_model(params)


if __name__ == "__main__":
    main()
