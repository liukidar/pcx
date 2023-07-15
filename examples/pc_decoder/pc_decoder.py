import os
import gc
from typing import Callable, Any
from argparse import ArgumentParser
from collections import defaultdict
from uuid import uuid4
import json
from pathlib import Path

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
from ray.air.config import RunConfig, ScalingConfig
from ray.air.result import Result
from ray.air.integrations.wandb import WandbLoggerCallback
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb

import seaborn as sns
import matplotlib.pyplot as plt

from params import Params, ModelParams

# Environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class PCGenerator(px.EnergyModule):
    def __init__(
        self,
        *,
        params: ModelParams,
        act_fn: Callable[[jax.Array], jax.Array] = jax.nn.gelu,
        internal_init_fn: Callable[[pxc.RandomKeyGenerator, int], jax.Array],
    ) -> None:
        super().__init__()

        self.internal_dim = params.internal_dim
        self.hidden_dim = params.hidden_dim
        self.output_dim = params.output_dim
        self.num_hidden_layers = params.num_hidden_layers
        self.act_fn = act_fn

        def internal_node_init_fn(node: px.Node, rkg: pxc.RandomKeyGenerator) -> None:
            value = internal_init_fn(rkg, self.internal_dim)
            node.set_activation("u", value)
            node.x.value = value

        self.fc_layers = (
            [nn.Linear(self.internal_dim, self.hidden_dim)]
            + [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_hidden_layers)]
            + [nn.Linear(self.hidden_dim, self.output_dim)]
        )
        self.pc_nodes = [px.Node(init_fn=internal_node_init_fn)] + [px.Node() for _ in range(self.num_layers)]

        self.pc_nodes[-1].x.frozen = True

    def __call__(self, example: jax.Array = None, internal_state: jax.Array = None) -> jax.Array:
        if internal_state is None:
            internal_state = self.internal_state  # this might be None as well.
        # Call the internal layer so it can initialize the internal state by calling init_fn.
        x = self.pc_nodes[0](internal_state)["x"]

        for i in range(self.num_layers):
            # No activation function at the last layer
            act_fn = self.act_fn if i < self.num_layers - 1 else lambda x: x
            x = self.pc_nodes[i + 1](act_fn(self.fc_layers[i](x)))["x"]

        # During training, fix target to the input
        # so that the energy encodes the difference between the prediction u and the target x.
        if example is not None:
            self.pc_nodes[-1]["x"] = example

        return self.prediction

    def predict(self, x: jax.Array | None = None) -> jax.Array:
        if x is None:
            x = self.internal_state
        for i in range(self.num_layers):
            # No activation function at the last layer
            act_fn = self.act_fn if i < self.num_layers - 1 else lambda x: x
            x = act_fn(self.fc_layers[i](x))
        return x

    @property
    def num_layers(self):
        return self.num_hidden_layers + 2

    @property
    def prediction(self):
        # Return the output ("u" is equal to "x" if the target is not fixed,
        # while it is the actual output of the model if the target is fixed)
        return self.pc_nodes[-1]["u"]

    @property
    def internal_state(self):
        return self.pc_nodes[0]["x"]

    def get_x_parameters(self) -> pxc.ParamDict:
        return self.parameters().filter(px.f(px.NodeParam)(frozen=False))

    def get_w_parameters(self) -> pxc.ParamDict:
        return self.parameters().filter(px.f(px.LayerParam))

    def save_weights(self, savedir: str) -> None:
        os.makedirs(savedir, exist_ok=True)

        weights = {}
        id_to_name = {}
        for param_name, param_value in self.get_w_parameters().items():
            param_id = str(uuid4())
            weights[param_id] = param_value.value
            id_to_name[param_id] = param_name

        jax.numpy.savez(os.path.join(savedir, "w_params.npz"), **weights)
        with open(os.path.join(savedir, "w_params_id_to_name.json"), "w") as outfile:
            json.dump(id_to_name, outfile, indent=4)

    def load_weights(self, savedir: str) -> None:
        with open(os.path.join(savedir, "w_params_id_to_name.json"), "r") as infile:
            id_to_name = json.load(infile)

        with np.load(os.path.join(savedir, "w_params.npz")) as npzfile:
            weights: dict[str, jax.Array] = {
                id_to_name[param_id]: jax.numpy.array(npzfile[param_id]) for param_id in npzfile.files
            }

            missing_parameters = set()
            for param_name, param in self.get_w_parameters().items():
                if param_name in weights:
                    if param.value.shape != weights[param_name].shape:
                        raise ValueError(
                            f"Parameter {param_name} has shape {param.value.shape} but loaded weight has shape {weights[param_name].shape}"
                        )
                    param.value = weights[param_name]
                else:
                    missing_parameters.add(param_name)

        if missing_parameters:
            print(f"ERROR: When loadings weights {len(missing_parameters)} were not found: {missing_parameters}")


def internal_init(rkg: pxc.RandomKeyGenerator, internal_dim: int) -> jax.Array:
    value = jax.random.normal(rkg(), (internal_dim,))
    return value


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


def download_datasets(params: Params):
    datasets.MNIST(root=params.data_dir, train=True, download=True)  # type: ignore
    datasets.MNIST(root=params.data_dir, train=False, download=True)  # type: ignore


def get_data_loaders(params: Params) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, float, float]:
    train_dataset = datasets.MNIST(root=params.data_dir, train=True)

    train_data = train_dataset.data / 255
    train_data_mean = torch.mean(train_data).item()
    train_data_std = torch.std(train_data).item()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((train_data_mean,), (train_data_std,)),
        ]
    )

    train_dataset = datasets.MNIST(root=params.data_dir, train=True, transform=transform)  # type: ignore
    test_dataset = datasets.MNIST(root=params.data_dir, train=False, transform=transform)  # type: ignore

    assert (
        len(train_dataset) % params.batch_size == len(test_dataset) % params.batch_size == 0
    ), "All batches must have the same size!"

    def collate_into_jax_arrays(examples: list[tuple[torch.Tensor, Any]]) -> tuple[jax.Array, list[Any]]:
        data = jax.numpy.array([x[0].numpy() for x in examples]).reshape(len(examples), -1)
        targets = [x[1] for x in examples]
        return data, targets

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=collate_into_jax_arrays,
        # drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=collate_into_jax_arrays,
        # drop_last=True,
    )

    return train_loader, test_loader, train_data_mean, train_data_std


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
    predictions = model.predict()
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
    predictions = model.predict()
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


def restore_image(x: jax.Array, train_data_mean: float, train_data_std: float) -> NDArray:
    x = x.reshape(28, 28)
    x = x * train_data_std + train_data_mean
    x = x * 255
    x = x.astype(np.uint8)
    return x


def save_prediction_as_png(
    x: jax.Array, save_path: str, title: str, train_data_mean: float, train_data_std: float
) -> None:
    x = restore_image(x, train_data_mean, train_data_std)
    plt.clf()
    plt.imshow(x, cmap="gray")
    plt.title(title)
    plt.savefig(save_path)


def train_model(params: Params) -> None:
    results_dir = Path(params.result_dir) / params.experiment_name
    if results_dir.exists() and any(results_dir.iterdir()):
        raise RuntimeError(f"Results dir {results_dir} already exists and is not empty!")
    results_dir.mkdir(parents=True, exist_ok=True)

    model = PCGenerator(
        params=params,
        act_fn=jax.nn.tanh,
        internal_init_fn=internal_init,
    )

    if params.load_weights_from is not None:
        model.load_weights(params.load_weights_from)

    # run = wandb.init(
    #     project="pc-decoder",
    #     name=params.experiment_name,
    #     tags=["predictive-coding", "autoencoder", "decoder"],
    #     config=params.to_dict(),
    # )

    with pxu.train(model, jax.numpy.zeros((params.batch_size, params.output_dim))):
        optim_x = pxu.Optim(
            optax.chain(optax.add_decayed_weights(weight_decay=params.optim_x_l2), optax.sgd(params.optim_x_lr)),
            model.get_x_parameters(),
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
            model.get_w_parameters(),
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

    train_loader, test_loader, train_data_mean, train_data_std = get_data_loaders(params)

    train_mses = []
    test_mses = []
    best_test_mse = float("inf")

    test_images = defaultdict(list)
    for examples, labels in test_loader:
        for example, label in zip(examples, labels):
            if len(test_images[label]) >= 10:
                continue
            test_images[label].append(example)

    test_examples = jax.numpy.concatenate(np.array([v for v in test_images.values()]))
    test_labels = jax.numpy.concatenate(np.array([[l] * len(v) for l, v in test_images.items()]))

    with tqdm(range(params.epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Train Epoch {epoch + 1}")

            epoch_train_mses = []
            with tqdm(train_loader, unit="batch") as tbatch:
                for examples, _ in tbatch:
                    tbatch.set_description(f"Train Batch {tbatch.n + 1}")

                    metric = train_batch_fn(examples)
                    epoch_train_mses.append(metric)
                    tbatch.set_postfix(mse=metric)
            epoch_train_mse = np.mean(epoch_train_mses[-params.use_n_batches_to_compute_metrics :])
            train_mses.append(epoch_train_mse)

            epoch_test_mses = []
            with tqdm(test_loader, unit="batch") as tbatch:
                for examples, _ in tbatch:
                    tbatch.set_description(f"Test Batch {tbatch.n + 1}")

                    metric = test_batch_fn(examples)
                    epoch_test_mses.append(metric)
                    tbatch.set_postfix(mse=metric)
            epoch_test_mse = np.mean(epoch_test_mses)
            test_mses.append(epoch_test_mse)

            epoch_report = {
                "epoch": epoch + 1,
                "train_mse": epoch_train_mse,
                "test_mse": epoch_test_mse,
            }

            should_save_intermediate_results = (
                params.save_intermediate_results and (epoch + 1) % params.save_results_every_n_epochs == 0
            )
            should_save_best_results = params.save_best_model and epoch_test_mse < best_test_mse
            if should_save_intermediate_results or should_save_best_results:
                print(
                    f"Saving results for epoch {epoch + 1}. Best epoch: {should_save_best_results}. MSE: {epoch_test_mse}"
                )
                epoch_results = results_dir / f"epochs_{epoch + 1}"
                epoch_results.mkdir()

                if should_save_best_results:
                    best_test_mse = epoch_test_mse
                    (results_dir / "best").symlink_to(epoch_results, target_is_directory=True)

                model.save_weights(epoch_results)
                with open(os.path.join(epoch_results, "report.json"), "w") as outfile:
                    json.dump(epoch_report, outfile, indent=4)

                internal_states = get_internal_states_on_batch(
                    examples=test_examples,
                    model=model,
                    optim_x=optim_x,
                    loss=loss,
                    T=params.T,
                )

                predictions = model.predict(internal_states)

                for i, (example, prediction, label) in enumerate(zip(test_examples, predictions, test_labels)):
                    fig, axes = plt.subplots(1, 2)
                    axes[0].imshow(restore_image(example, train_data_mean, train_data_std), cmap="gray")
                    axes[0].set_title("Original")
                    axes[1].imshow(restore_image(prediction, train_data_mean, train_data_std), cmap="gray")
                    axes[1].set_title("Prediction")
                    # Set figure title
                    fig.suptitle(f"Epoch {epoch + 1} Label {label} Example {i}")
                    fig.savefig(epoch_results / f"label_{label}_example_{i}.png")

            wandb.log(epoch_report)
            session.report(epoch_report)

            tepoch.set_postfix(train_mse=epoch_train_mse, test_mse=epoch_test_mse)

    # result_dir = pathlib.Path(params.result_dir)
    # result_dir.mkdir(parents=True, exist_ok=True)

    # fig, ax = plt.subplots()
    # sns.lineplot(x=range(len(train_mses)), y=train_mses, ax=ax, label="Train MSE avg")
    # sns.lineplot(x=range(len(test_mses)), y=test_mses, ax=ax, label="Test MSE avg")
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("MSE avg per epoch")
    # ax.set_title("MSE avg per epoch")
    # ax.legend()
    # fig.savefig(result_dir / "pc_decoder_mnist.png")

    # internal_states_data = {
    #     "x": [],
    #     "y": [],
    #     "label": [],
    # }

    # with tqdm(test_loader, unit="batch") as tbatch:
    #     for examples, labels in tbatch:
    #         tbatch.set_description(f"Test Batch {tbatch.n + 1}")

    #         states = get_internal_states_fn(examples)
    #         assert states.shape == (params.batch_size, params.internal_dim)
    #         assert labels.shape == (params.batch_size,)
    #         internal_states_data["x"].extend(states[:, 0].tolist())
    #         internal_states_data["y"].extend(states[:, 1].tolist())
    #         internal_states_data["label"].extend(labels.tolist())

    # internal_states_df = pd.DataFrame(internal_states_data)

    # plt.clf()
    # sns.scatterplot(data=internal_states_df, x="x", y="y", hue="label")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("Internal representations of classes")
    # plt.legend()
    # plt.savefig(result_dir / "pc_decoder_mnist_internal_states.png")


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
    download_datasets(params)

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
