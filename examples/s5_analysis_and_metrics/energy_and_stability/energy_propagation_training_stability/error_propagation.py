import functools
import os
import random
from typing import Any, Callable, Dict, List
import pickle
import multiprocessing

from omegaconf import DictConfig
from tqdm import tqdm
import hydra
import jax

import utils

cluster = utils.cluster.ClusterManager()


def single_trial_wrapper(variable_kwargs: dict, constant_kwargs: dict) -> Callable:
    return utils.single_trial(**constant_kwargs, **variable_kwargs)


def validate_cfg(cfg: DictConfig):
    if cfg.data.dataset not in ["mnist", "fashion_mnist"]:
        assert not cfg.data.resize.enabled, f"Data resizing is only supported for MNIST and FashionMNIST. Got {cfg.data.dataset}."

    assert (
        not cfg.run.reload_data
    ), "Reloading data is not supported for this experiment."
    assert (
        not cfg.model.constant_layer_size
    ), "Constant layer size is not supported for this experiment."


def create_experimental_conditions(
    cfg: DictConfig, shuffle: bool = True
) -> List[Dict[str, Any]]:
    combined_pars = list(
        utils.product_dict(
            hidden_dims=cfg.experiment.h_dims,
            h_lr=cfg.experiment.h_lr,
            w_lr=cfg.experiment.w_lr,
            optimizer_w=cfg.experiment.optimizer_w,
            momentum_w=cfg.experiment.momentum_w,
            act_fn=cfg.experiment.activation_fn,
            seed=list(range(cfg.experiment.seeds)),
        )
    )
    # remove adamw with momentum != 0.9
    combined_pars = [
        c
        for c in combined_pars
        if not (c["optimizer_w"] == "adamw" and c["momentum_w"] != 0.9)
    ]
    if shuffle:
        random.shuffle(combined_pars)

    print(f"Number of experimental conditions: {len(combined_pars)}")
    return combined_pars


@hydra.main(config_path="config", config_name="error_propagation", version_base="1.3")
def main(cfg: DictConfig):
    validate_cfg(cfg)

    # disable jit for debugging
    jax.config.update("jax_disable_jit", (not cfg.run.jit))

    # create experimental conditions
    combined_pars = create_experimental_conditions(cfg)

    # experiment name using uuid
    experiment_name, experiment_folder = utils.setup_experiment_log(
        cfg, "error_propagation"
    )

    # get number of cores to use
    n_parallel = utils.set_number_of_cores(cfg.run.n_parallel)

    # get data function
    get_data = functools.partial(
        utils.data.get_data,
        dataset=cfg.data.dataset,
        num_samples=cfg.data.num_samples,
        batch_size=cfg.data.batch_size,
        num_epochs=cfg.optim.num_epochs,
        resize_enabled=cfg.data.resize.enabled,
        resize_method=cfg.data.resize.method,
    )

    # get data for training and evaluation
    (X, y), (X_test, y_test), train_dl, test_dl, input_dim, output_dim, num_epochs = (
        get_data(num_classes=cfg.data.num_classes, resize_size=cfg.data.resize.size)
    )

    # set up data arguments passed to the trial
    reload_data = get_data if cfg.run.reload_data else None

    # Create partial function for trials with parameters that remain constant across all calls.
    single_trial_partial = functools.partial(
        single_trial_wrapper,
        constant_kwargs=dict(
            batch_size=cfg.data.batch_size,
            num_epochs=num_epochs,
            reload_data=reload_data,
            train_dl=train_dl,
            test_dl=test_dl,
            model_definition=cfg.model.definition,
            input_dim=input_dim,
            output_dim=output_dim,
            T=cfg.optim.h.T,
            init_h=cfg.model.init_h,
            init_h_sd=cfg.model.init_h_sd,
            init_w=cfg.model.init_w,
            constant_layer_size=cfg.model.constant_layer_size,
        ),
    )

    # Execute the model training in parallel if n_parallel > 1, otherwise sequentially
    if n_parallel == 1:
        results = []
        for par in tqdm(combined_pars, desc=f"Using {n_parallel} processes"):
            results.append(single_trial_partial(par))
    else:
        with multiprocessing.get_context("spawn").Pool(processes=n_parallel) as pool:
            results = []

            for result in tqdm(
                pool.imap_unordered(single_trial_partial, combined_pars),
                total=len(combined_pars),
                desc=f"Using {n_parallel} processes",
                smoothing=0.1,
            ):
                results.append(result)

    # save results
    results = utils.jax_tree_to_numpy(results)
    save_path = os.path.join(experiment_folder, "results.pickle")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {save_path}")
    # get mb of results
    print(f"Results size: {os.path.getsize(save_path) / 1e6:.2f} MB")
    print(f"Finished experiment {experiment_name}")


if __name__ == "__main__":
    main()
