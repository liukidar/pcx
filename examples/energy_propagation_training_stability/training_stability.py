import functools
import os
import pickle
import random
from typing import Callable
import multiprocessing

import jax
from omegaconf import DictConfig
from tqdm import tqdm
import hydra

import utils


cluster = utils.cluster.ClusterManager()


def single_trial_wrapper(variable_kwargs: dict, constant_kwargs: dict) -> Callable:
    return utils.single_trial(**constant_kwargs, **variable_kwargs)


def validate_cfg(cfg: DictConfig) -> None:
    """
    Validates configuration dictionary.

    This function checks the consistency of the input configuration dictionary.
    It raises assertions if certain conditions are not met.

    Args:
        cfg (DictConfig): The configuration dictionary to be validated.

    Returns:
        None

    Raises:
        AssertionError: If the configuration does not meet certain requirements.

    Notes:
        * constant_layer_size requires reloading data for each trial and overwriting `cfg.data.num_classes` and `cfg.data.resize`.
        * Data resizing is only supported for MNIST and FashionMNIST datasets.
    """

    if cfg.model.constant_layer_size:
        print(
            "constant layer size selected. This reloads the data for each trial and overwrites cfg.data.num_classes and cfg.data.resize."
        )
        assert (
            cfg.run.reload_data
        ), "constant layer size require reloading data for each trial."
        assert (
            cfg.data.resize.enabled
        ), "Data resizing should be enabled for constant layer size."

    if cfg.data.resize.enabled:
        assert (
            cfg.data.dataset
            in [
                "mnist",
                "fashion_mnist",
            ]
        ), f"Data resizing is only supported for MNIST and FashionMNIST. Got {cfg.data.dataset}."

    if cfg.model.definition == "BP":
        assert cfg.optim.h.T == 1, "When using BP, h is not optimised. Choose T=1."


@hydra.main(config_path="config", config_name="training_stability", version_base="1.3")
def main(cfg):
    validate_cfg(cfg)

    # disable jit for debugging
    jax.config.update("jax_disable_jit", (not cfg.run.jit))

    # parameters: We test over the following values
    h_lrs = [
        s * lr for s in cfg.experiment.h_lr_scalars for lr in cfg.experiment.h_lr_steps
    ]
    combined_pars = list(
        utils.product_dict(
            hidden_dims=cfg.experiment.h_dims,
            h_lr=h_lrs,
            seed=list(range(cfg.experiment.seeds)),
        )
    )
    random.shuffle(combined_pars)

    # experiment name using uuid
    experiment_name, experiment_folder = utils.setup_experiment_log(
        cfg, "training_stability"
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
            optimizer_w=cfg.optim.w.optimizer,
            momentum_w=cfg.optim.w.momentum,
            model_definition=cfg.model.definition,
            input_dim=input_dim,
            output_dim=output_dim,
            T=cfg.optim.h.T,
            act_fn=cfg.model.activation,
            init_h=cfg.model.init_h,
            init_h_sd=cfg.model.init_h_sd,
            init_w=cfg.model.init_w,
            w_lr=cfg.optim.w.lr,
            constant_layer_size=cfg.model.constant_layer_size,
        ),
    )
    # Execute the model training in parallel if n_cores > 1, otherwise sequentially
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

    # postprocess results
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
