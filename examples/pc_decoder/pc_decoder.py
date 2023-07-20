import os
import gc
from typing import Callable, Any
from argparse import ArgumentParser
from collections import defaultdict
from uuid import uuid4
import json
from pathlib import Path
from datetime import datetime

# from functools import partial
import pprint

import jax
import jax.numpy as jnp
import optax
import pcax as px  # type: ignore
import pcax.utils as pxu  # type: ignore
import pcax.nn as nn  # type: ignore
import pcax.core as pxc  # type: ignore
import torch
from torch.utils.data import Dataset, DataLoader  # type: ignore
from ray.air import session
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb
import umap

import seaborn as sns
import matplotlib.pyplot as plt

from params import Params, ModelParams

# Environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


DEBUG = os.environ.get("DEBUG", "0") == "1"
DEBUG_BATCH_NUMBER = 10


activation_functions: dict[str, Callable[[jax.Array], jax.Array]] = {
    "gelu": jax.nn.gelu,
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
}


class PCGenerator(px.EnergyModule):
    def __init__(
        self,
        *,
        params: ModelParams,
        internal_init_fn: Callable[[pxc.RandomKeyGenerator, int], jax.Array],
    ) -> None:
        super().__init__()

        self.internal_dim = params.internal_dim
        self.hidden_dim = params.hidden_dim
        self.output_dim = params.output_dim
        self.num_hidden_layers = params.num_hidden_layers
        self.act_fn = activation_functions[params.activation]

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

    def __call__(self, example: jax.Array | None = None, internal_state: jax.Array | None = None) -> jax.Array:
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

    def feed_forward_predict(self, x: jax.Array | None = None) -> jax.Array:
        if x is None:
            x = self.internal_state
        if x is None:
            raise RuntimeError("Internal state is none.")
        for i in range(self.num_layers):
            # No activation function at the last layer
            act_fn = self.act_fn if i < self.num_layers - 1 else lambda x: x
            x = act_fn(self.fc_layers[i](x))
        return x

    def converge_on_batch(self, examples: jax.Array, *, optim_x, loss, T) -> None:
        grad_and_values = pxu.grad_and_values(
            px.f(px.NodeParam)(frozen=False),
        )(loss)
        with pxu.eval(self, examples):
            for i in range(T):
                with pxu.step(self):
                    g, _ = grad_and_values(examples, model=self)

                    optim_x(g)

    @property
    def num_layers(self) -> int:
        return self.num_hidden_layers + 2

    @property
    def prediction(self) -> jax.Array:
        # Return the output ("u" is equal to "x" if the target is not fixed,
        # while it is the actual output of the model if the target is fixed)
        res = self.pc_nodes[-1]["u"]
        assert isinstance(res, jax.Array)
        return res

    @property
    def internal_state(self) -> jax.Array | None:
        res = self.pc_nodes[0]["x"]
        assert res is None or isinstance(res, jax.Array)
        return res

    def get_x_parameters(self) -> pxc.ParamDict:
        res = self.parameters().filter(px.f(px.NodeParam)(frozen=False))
        assert isinstance(res, pxc.ParamDict)
        return res

    def get_w_parameters(self) -> pxc.ParamDict:
        res = self.parameters().filter(px.f(px.LayerParam))
        assert isinstance(res, pxc.ParamDict)
        return res

    def save_weights(self, savedir: str) -> None:
        os.makedirs(savedir, exist_ok=True)

        weights = {}
        id_to_name = {}
        for param_name, param_value in self.get_w_parameters().items():
            param_id = str(uuid4())
            id_to_name[param_id] = param_name
            weights[param_id] = param_value.value

        jnp.savez(os.path.join(savedir, "w_params.npz"), **weights)
        with open(os.path.join(savedir, "w_params_id_to_name.json"), "w") as outfile:
            json.dump(id_to_name, outfile, indent=4)

    def load_weights(self, savedir: str) -> None:
        with open(os.path.join(savedir, "w_params_id_to_name.json"), "r") as infile:
            id_to_name = json.load(infile)

        with np.load(os.path.join(savedir, "w_params.npz")) as npzfile:
            weights: dict[str, jax.Array] = {
                id_to_name[param_id]: jnp.array(npzfile[param_id]) for param_id in npzfile.files
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
    # TODO: Play with different initialization strategies
    value = jax.random.normal(rkg(), (internal_dim,))
    return value


@pxu.vectorize(px.f(px.NodeParam, with_cache=True), in_axis=(0,), out_axis=("sum",))
def loss(example: jax.Array, *, model: PCGenerator) -> jax.Array:
    model(example=example)
    res = model.energy()
    assert isinstance(res, jax.Array)
    return res


@pxu.vectorize(px.f(px.NodeParam, with_cache=True), in_axis=(0,))
def predict(example: jax.Array, *, model) -> jax.Array:
    res = model(example=example)
    assert isinstance(res, jax.Array)
    return res


@pxu.vectorize(px.f(px.NodeParam, with_cache=True), in_axis=(0,))
def feed_forward_predict(internal_state: jax.Array, *, model: PCGenerator) -> jax.Array:
    res = model.feed_forward_predict(internal_state)
    assert isinstance(res, jax.Array)
    return res


def download_datasets(params: Params):
    datasets.MNIST(root=params.data_dir, train=True, download=True)  # type: ignore
    datasets.MNIST(root=params.data_dir, train=False, download=True)  # type: ignore


if DEBUG:
    from itertools import islice

    class ReentryIsliceIterator:
        def __init__(self, iterable, limit):
            self.iterable = iterable
            self.limit = limit

        def __iter__(self):
            return islice(self.iterable, self.limit).__iter__()

        def __getattr__(self, attr):
            return getattr(self.iterable, attr)


def get_data_loaders(params: Params) -> tuple[DataLoader[Any], DataLoader[Any], float, float]:
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
        data = jnp.array([x[0].numpy() for x in examples]).reshape(len(examples), -1)
        targets = [x[1] for x in examples]
        return data, targets

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=collate_into_jax_arrays,
        # drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=collate_into_jax_arrays,
        # drop_last=True,
    )

    if DEBUG:
        train_loader = ReentryIsliceIterator(train_loader, DEBUG_BATCH_NUMBER)  # type: ignore
        test_loader = ReentryIsliceIterator(test_loader, DEBUG_BATCH_NUMBER)  # type: ignore

    return train_loader, test_loader, train_data_mean, train_data_std


def get_viz_samples(test_loader: DataLoader[Any]) -> tuple[jax.Array, jax.Array]:
    num_classes = 10
    batch_size = test_loader.batch_size
    assert batch_size is not None
    num_per_label = batch_size // num_classes
    additional_examples = max(0, batch_size - num_per_label * num_classes)

    selected_indexes = defaultdict(list)
    for index, (_, label) in enumerate(test_loader.dataset):
        selected_indexes[label].append(index)
    for label in selected_indexes:
        additional_examples += max(0, num_per_label - len(selected_indexes[label]))
    for label in selected_indexes:
        selected_indexes[label] = selected_indexes[label][: num_per_label + additional_examples]
        if len(selected_indexes[label]) > num_per_label:
            additional_examples = 0
    selected_examples = []
    selected_labels = []
    for label, indexes in selected_indexes.items():
        for index in indexes:
            selected_examples.append(test_loader.dataset[index][0])
            assert test_loader.dataset[index][1] == label
            selected_labels.append(label)

    examples = jnp.array([x.reshape(-1).numpy() for x in selected_examples])
    labels = jnp.array(selected_labels)
    return examples, labels


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
def train_on_batch(examples: jax.Array, *, model: PCGenerator, optim_x, optim_w, loss, T) -> jax.Array:
    grad_and_values = pxu.grad_and_values(
        px.f(px.NodeParam)(frozen=False) | px.f(px.LayerParam),  # type: ignore
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
                # TODO: try PC instead of PPC here
                optim_x(g)
                optim_w(g)
    predictions = feed_forward_predict(model.internal_state, model=model)[0]
    mse = jnp.mean((predictions - examples) ** 2)
    return mse


@pxu.jit()
def test_on_batch(examples, *, model: PCGenerator, optim_x, loss, T) -> jax.Array:
    model.converge_on_batch(examples, optim_x=optim_x, loss=loss, T=T)
    predictions = feed_forward_predict(model.internal_state, model=model)[0]
    mse = jnp.mean((predictions - examples) ** 2)
    return mse


@pxu.jit()
def get_internal_states_on_batch(examples, *, model: PCGenerator, optim_x, loss, T) -> jax.Array:
    model.converge_on_batch(examples, optim_x=optim_x, loss=loss, T=T)
    assert model.internal_state is not None
    return model.internal_state


def restore_image(image_array: jax.Array, train_data_mean: float, train_data_std: float) -> NDArray[np.uint8]:
    x = np.asarray(image_array)
    x = x.reshape(28, 28)
    x = x * train_data_std + train_data_mean
    x = x * 255
    x = x.astype(np.uint8)
    return x


def train_model(params: Params) -> None:
    results_dir = Path(params.results_dir) / params.experiment_name
    if results_dir.exists() and any(results_dir.iterdir()):
        raise RuntimeError(f"Results dir {results_dir} already exists and is not empty!")
    results_dir.mkdir(parents=True, exist_ok=True)

    model = PCGenerator(
        params=params,
        internal_init_fn=internal_init,
    )

    if params.load_weights_from is not None:
        model.load_weights(params.load_weights_from)

    run_name = f"{params.experiment_name}--{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    if params.do_hypertunning:
        run_name += f"--{tune.get_trial_id()}"
    # run = wandb.init(
    #     project="pc-decoder",
    #     name=run_name,
    #     tags=["predictive-coding", "autoencoder", "decoder"],
    #     config=params.to_dict(),
    # )

    with pxu.train(model, jnp.zeros((params.batch_size, params.output_dim))):
        optim_x = pxu.Optim(
            optax.chain(
                optax.add_decayed_weights(weight_decay=params.optim_x_l2),
                optax.sgd(params.optim_x_lr / params.batch_size),
            ),
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

    viz_examples, viz_labels = get_viz_samples(test_loader=test_loader)

    with tqdm(range(params.epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Train Epoch {epoch + 1}")

            epoch_train_mses = []
            with tqdm(train_loader, unit="batch") as tbatch:
                for examples, _ in tbatch:
                    tbatch.set_description(f"Train Batch {tbatch.n + 1}")
                    metric = train_batch_fn(examples).item()
                    epoch_train_mses.append(metric)
                    tbatch.set_postfix(mse=metric)
            epoch_train_mse = np.mean(epoch_train_mses[-params.use_last_n_batches_to_compute_metrics :])
            train_mses.append(epoch_train_mse)

            epoch_test_mses = []
            with tqdm(test_loader, unit="batch") as tbatch:
                for examples, _ in tbatch:
                    tbatch.set_description(f"Test Batch {tbatch.n + 1}")
                    metric = test_batch_fn(examples).item()
                    epoch_test_mses.append(metric)
                    tbatch.set_postfix(mse=metric)
            epoch_test_mse = np.mean(epoch_test_mses)
            test_mses.append(epoch_test_mse)

            epoch_report = {
                "epochs": epoch + 1,
                "train_mse": epoch_train_mse,
                "test_mse": epoch_test_mse,
            }

            should_save_intermediate_results = (
                params.save_intermediate_results and (epoch + 1) % params.save_results_every_n_epochs == 0
            )
            should_save_best_results = params.save_best_results and epoch_test_mse < best_test_mse
            if should_save_intermediate_results or should_save_best_results:
                print(
                    f"Saving results for epoch {epoch + 1}. Best epoch: {should_save_best_results}. MSE: {epoch_test_mse}"
                )
                epoch_results = results_dir / f"epochs_{epoch + 1}"
                epoch_results.mkdir()

                if should_save_best_results:
                    best_test_mse = epoch_test_mse
                    (results_dir / "best").unlink(missing_ok=True)
                    (results_dir / "best").symlink_to(epoch_results.relative_to(results_dir), target_is_directory=True)

                model.save_weights(str(epoch_results))
                with open(os.path.join(epoch_results, "report.json"), "w") as outfile:
                    json.dump(epoch_report, outfile, indent=4)

                internal_states = get_internal_states_on_batch(
                    examples=viz_examples,  # FIXME: select test images in a stratified manner
                    model=model,
                    optim_x=optim_x,
                    loss=loss,
                    T=params.T,
                )

                predictions = feed_forward_predict(internal_states, model=model)[0]
                for i, (example, prediction, label) in enumerate(zip(viz_examples, predictions, viz_labels)):
                    fig, axes = plt.subplots(1, 2)
                    axes[0].imshow(restore_image(example, train_data_mean, train_data_std), cmap="gray")
                    axes[0].set_title("Original")
                    axes[1].imshow(restore_image(prediction, train_data_mean, train_data_std), cmap="gray")
                    axes[1].set_title("Prediction")
                    # Set figure title
                    fig.suptitle(f"Epoch {epoch + 1} Label {label} Example {i}")
                    fig.savefig(epoch_results / f"label_{label}_example_{i}.png")

                # TODO: configure UMAP
                reduced_internal_states = umap.UMAP().fit_transform(jnp.asarray(internal_states))

                internal_states_data: dict[str, list[float]] = {
                    "x": [],
                    "y": [],
                    "label": [],
                }
                internal_states_data["x"].extend(reduced_internal_states[:, 0].tolist())
                internal_states_data["y"].extend(reduced_internal_states[:, 1].tolist())
                internal_states_data["label"].extend(viz_labels.tolist())
                internal_states_df = pd.DataFrame(internal_states_data).sort_values("label")

                plt.clf()
                sns.scatterplot(data=internal_states_df, x="x", y="y", hue="label")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.title("Internal representations of classes")
                plt.legend()
                plt.savefig(epoch_results / "pc_decoder_mnist_internal_states.png")

            # wandb.log(epoch_report)
            session.report(epoch_report)

            tepoch.set_postfix(train_mse=epoch_train_mse, test_mse=epoch_test_mse)


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

    if DEBUG:
        import shutil

        params.experiment_name = "debug"
        results_dir = os.path.join(params.results_dir, params.experiment_name)
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        params.epochs = 2
        params.batch_size = 10
        params.T = 4
        params.use_last_n_batches_to_compute_metrics = 5
        params.save_best_results = True
        params.save_intermediate_results = True
        params.save_results_every_n_epochs = 1

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
                local_dir=params.results_dir,
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
