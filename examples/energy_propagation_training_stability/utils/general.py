import multiprocessing
from typing import Callable, Iterable, Tuple
import os
import uuid
import math
import random
import itertools

from omegaconf import DictConfig, OmegaConf
import numpy as np

import jax
import jax.numpy as jnp
import optax
from jax.tree_util import tree_map

import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.functional as pxf
import pcax.utils as pxu

from utils.models import Model, ModelBP, ModelPC
from utils.cluster import ClusterManager


cluster = ClusterManager()


# thank you https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def set_number_of_cores(n_cores: int) -> int:
    """Set the number of cores to use for parallel processing.
    If n_cores is positive, it will be used as the number of cores to use.
    If n_cores is negative, all (automatically detected) will be used, minus ``n_cores``.

    Parameters:
        n_cores: int
            The number of cores to use for parallel processing.

    Returns:
        int
            The number of cores to use for parallel processing.
    """
    if n_cores > 0:
        n_cores = n_cores
    else:
        n_cores = multiprocessing.cpu_count() + n_cores
    return n_cores


def int_root(n: float | int) -> int:
    return int(math.sqrt(n))


def setup_experiment_log(cfg: DictConfig, study_name: str) -> Tuple[str, str]:
    """Setup experiment logging.

    Sets up an experiment log with a unique name and folder. This function creates a new folder in the artifact directory based on the configuration and logs the experiment details to the console and to a YAML file.
    It generates a unique experiment name using UUIDs, creates a folder for the experiment in the artifact directory, writes the experiment configuration to the console and to a YAML file, and returns the experiment name and folder path.

    Parameters:
        cfg: DictConfig
            The experiment configuration.

    Returns:
        Tuple[str, str]
            A tuple containing the experiment name and folder path.
    """
    experiment_name = f"{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}"
    experiment_folder = os.path.join(cluster.artifact_dir, study_name, experiment_name)
    print("-" * 80)
    print("Config")
    print("-" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("-" * 80)
    print(f"Experiment name: {experiment_name}")
    os.makedirs(experiment_folder, exist_ok=True)
    print(f"Experiment folder: {experiment_folder}")
    print("-" * 80)
    OmegaConf.save(cfg, os.path.join(experiment_folder, "config.yaml"))
    return experiment_name, experiment_folder


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: Model):
    return model(x, y)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: Model):
    y_ = model(x, None)
    return jax.lax.pmean(model.energy().sum(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch(T: int, x: jax.Array, y: jax.Array, *, model: Model, optim_w: pxu.Optim, optim_h: pxu.Optim):
    model.train()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)

    energies_per_layer_list = []
    for i in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, y_), g = pxf.value_and_grad(pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]), has_aux=True)(energy)(
                x, model=model
            )
            energies_per_layer = {str(layer_idx): value_node.energy() for layer_idx, value_node in enumerate(model.vodes)}
            energies_per_layer_list.append(energies_per_layer)

        if optim_h is not None:
            optim_h.step(model, g["model"], True)

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, y_), g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(energy)(x, model=model)

    optim_w.step(model, g["model"])

    # weight and grad norms
    w_grad_norms_per_layer = {str(k): jnp.linalg.norm(v.nn.weight.get()) for k, v in enumerate(g["model"].layers)}
    w_norms_per_layer = {str(k): jnp.linalg.norm(v.nn.weight.get()) for k, v in enumerate(model.layers)}
    w_grad_vars_per_layer = {str(k): jnp.var(v.nn.weight.get()) for k, v in enumerate(g["model"].layers)}

    return {
        "energies": energies_per_layer_list[-1],
        "w_grad_norms": w_grad_norms_per_layer,
        "w_norms": w_norms_per_layer,
        "w_grad_vars": w_grad_vars_per_layer,
    }


@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: Model):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model)

    y_pred = jnp.argmax(y_, axis=-1)
    y = jnp.argmax(y, axis=-1)

    return (y_pred == y).sum(), y_pred


def train(dl, T, *, model: Model, optim_w: pxu.Optim, optim_h: pxu.Optim):
    return_dicts = []
    for x, y in dl:
        return_dict = train_on_batch(T, x, y, model=model, optim_w=optim_w, optim_h=optim_h)
        return_dicts.append(return_dict)

    # reshape return_dicts
    # First: List[Dict[]] to Dict[List[]]
    return_dicts = {k: [d[k] for d in return_dicts] for k in return_dicts[0]}
    # Second: stack lists
    for metric_name, metric in return_dicts.items():
        return_dicts[metric_name] = {k: jnp.stack([v[k] for v in metric]) for k in metric[0].keys()}

    # get norm and variance of energies
    return_dicts["energy_norms"] = {k: jnp.linalg.norm(v, axis=-1) for k, v in return_dicts["energies"].items()}
    return_dicts["energy_vars"] = {k: jnp.var(v, axis=-1) for k, v in return_dicts["energies"].items()}
    del return_dicts["energies"]

    return return_dicts


def eval(dl, *, model: Model):
    n_correct = 0
    n_total = 0
    ys_ = []

    for x, y in dl:
        c, y_ = eval_on_batch(x, y, model=model)
        n_correct += c
        n_total += x.shape[0]
        ys_.append(y_)

    return jnp.array(n_correct / n_total), np.concatenate(ys_)


# we merge initialisation in a single function to be able to create multiple models.
def init(
    model_definition: str,
    batch_size: int,
    input_dim: int,
    hidden_dims: int,
    output_dim: int,
    h_lr: float,
    init_h: str,
    init_h_sd: float,
    num_layers: int,
    act_fn: str,
    w_lr: float,
    init_w: str,
    optimizer_w: str,
    momentum_w: float,
    random_key_generator: px.RandomKeyGenerator = None,
):
    # check consistency for hidden dim
    if not isinstance(hidden_dims, int):
        assert (
            len(hidden_dims) == num_layers - 2
        ), f"Number of hidden dims should be equal to the number of hidden layers - 2. Got {len(hidden_dims)} and {nm_layers - 2}."

    # model definition: PC or BP
    if model_definition == "PC":
        ModelClass = ModelPC
    elif model_definition == "BP":
        ModelClass = ModelBP
    else:
        raise ValueError(f"Unknown model definition: {model_definition}")

    # activation function
    act_fn_list = {
        "relu": jax.nn.relu,
        "tanh": jax.nn.tanh,
        "sigmoid": jax.nn.sigmoid,
        "hard_tanh": jax.nn.hard_tanh,
        "leaky_relu": jax.nn.leaky_relu,
    }
    assert act_fn in act_fn_list, f"Unknown activation function: {act_fn}"
    act_fn = act_fn_list[act_fn]

    # initialise model
    model = ModelClass(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        num_layers=num_layers,
        act_fn=act_fn,
        random_key_generator=random_key_generator,
        init_w=init_w,
        init_h=init_h,
        init_h_sd=init_h_sd,
    )

    # initialise optimisers
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jax.numpy.zeros((batch_size, input_dim)), None, model=model)
        if model_definition == "PC":
            optim_h = pxu.Optim(optax.sgd(h_lr), pxu.Mask(pxc.VodeParam)(model))
        else:
            optim_h = None
        if optimizer_w == "sgd":
            optim_w = pxu.Optim(optax.sgd(w_lr, momentum=momentum_w), pxu.Mask(pxnn.LayerParam)(model))
        elif optimizer_w == "adamw":
            optim_w = pxu.Optim(optax.adamw(w_lr), pxu.Mask(pxnn.LayerParam)(model))
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_w}")

    return model, optim_h, optim_w


def run(nm_epochs, model, optim_h, optim_w, train_dl, test_dl, T):
    train_metrics_list = []
    for _ in range(nm_epochs):
        random.shuffle(train_dl)
        train_metrics = train(train_dl, T=T, model=model, optim_w=optim_w, optim_h=optim_h)
        train_metrics_list.append(train_metrics)
    a, _ = eval(test_dl, model=model)

    # stack train metrics
    # First: List[Dict[]] to Dict[List[]]
    train_metrics_list = {k: [d[k] for d in train_metrics_list] for k in train_metrics_list[0]}
    # Second: stack lists
    for metric_name, metric in train_metrics_list.items():
        train_metrics_list[metric_name] = {k: jnp.stack([v[k] for v in metric]) for k in metric[0].keys()}

    return {"accuracy": a, **train_metrics_list}


def single_trial(
    # params: Tuple[float | Tuple[float], float],
    batch_size: int,
    num_epochs: int,
    reload_data: Callable | None,
    train_dl: Iterable | None,
    test_dl: Iterable | None,
    optimizer_w: str,
    momentum_w: float,
    model_definition: str,
    input_dim: int | None,
    hidden_dims: int | Tuple[int],
    output_dim: int | None,
    act_fn: str,
    T: int,
    init_h: str,
    init_h_sd: float,
    h_lr: float,
    init_w: str,
    w_lr: float,
    constant_layer_size: bool,
    seed: int,
    verbose: bool = True,
) -> float:
    # extract params into seperate variables
    # h_dims, h_lr, seed = params
    seed = int(seed)

    # convert h_dims to int. Either Tuple[Any] -> Tuple[int] or Any -> int
    try:
        h_dims = [int(h) for h in hidden_dims]
    except TypeError:
        h_dims = int(hidden_dims)

    # if constant_layer_size is selected, the h_dims sets the model size. num_classes and image size will be overwritting.
    if constant_layer_size:
        assert isinstance(h_dims, int), "Only uniform hidden layer sizes (int) supported for constant_layer_size."
        input_dim = output_dim = h_dims
        if verbose:
            print(
                f"constant_layer_size selected. Using hidden layer size {h_dims}. Overwriting: num_classes={output_dim} and resize_size={int_root(input_dim)}."
            )

    # if reload_data is a function, it will be used to reload data with possibly overwritten parameters
    if reload_data is not None:
        (X, y), (X_test, y_test), train_dl, test_dl, input_dim, output_dim, num_epochs = reload_data(
            num_classes=output_dim, resize_size=int_root(input_dim)
        )

    # initialise model and optimizers
    RKG = px.RandomKeyGenerator(0)
    model, optim_h, optim_w = init(
        model_definition=model_definition,
        batch_size=batch_size,
        input_dim=input_dim,
        hidden_dims=h_dims,
        output_dim=output_dim,
        h_lr=h_lr,
        init_h=init_h,
        init_h_sd=init_h_sd,
        num_layers=4,
        act_fn=act_fn,
        w_lr=w_lr,
        init_w=init_w,
        optimizer_w=optimizer_w,
        momentum_w=momentum_w,
        random_key_generator=RKG,
    )

    results = run(num_epochs, model, optim_h, optim_w, train_dl, test_dl, T)

    # run trial and return
    return {
        "config": {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "reload_data": reload_data is not None,
            "optimizer_w": optimizer_w,
            "momentum_w": momentum_w,
            "model_definition": model_definition,
            "input_dim": input_dim,
            "hidden_dims": h_dims,
            "output_dim": output_dim,
            "act_fn": act_fn,
            "T": T,
            "init_h": init_h,
            "init_h_sd": init_h_sd,
            "h_lr": h_lr,
            "init_w": init_w,
            "w_lr": w_lr,
            "constant_layer_size": constant_layer_size,
            "seed": seed,
        },
        "results": {**results},
    }


def jax_tree_to_numpy(tree: dict | list) -> dict | list:
    """Converts a JAX tree to numpy tree."""

    def convert(x):
        if isinstance(x, jnp.ndarray):
            return np.array(x)
        else:
            return x

    return tree_map(convert, tree)
