import gc
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
import wandb
from pc_decoder.data_loading import get_data_loaders, get_stratified_test_batch
from pc_decoder.logging import (
    init_wandb,
    log_test_t_step_metrics,
    log_train_batch_metrics,
    log_train_t_step_metrics,
)
from pc_decoder.model import PCDecoder, feed_forward_predict, model_energy_loss
from pc_decoder.params import Params
from pc_decoder.visualization import create_all_visualizations
from ray import tune
from ray.air import session
from tqdm import tqdm  # type: ignore

import pcax.utils as pxu  # type: ignore

DEBUG = os.environ.get("DEBUG", "0") == "1"
DEBUG_BATCH_NUMBER = 10

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

if DEBUG:
    from itertools import islice

    class ReentryIsliceIterator:
        def __init__(self, iterable, limit):
            self.iterable = iterable
            self.limit = limit

        def __iter__(self):
            return iter(islice(self.iterable, self.limit))

        def __len__(self):
            return min(self.limit, len(self.iterable))

        def __getattr__(self, attr):
            return getattr(self.iterable, attr)


def internal_state_init(
    params: Params,
    prng_key: jax.random.KeyArray,
) -> tuple[jax.Array, jax.random.KeyArray]:
    # TODO: Play with different initialization strategies
    value = jnp.zeros((params.internal_dim,))
    return value, prng_key


@pxu.jit()
def train_on_batch(
    examples: jax.Array,
    *,
    model: PCDecoder,
    optim_x: pxu.Optim,
    optim_w: pxu.Optim,
    loss_fn: Callable,
) -> tuple[jax.Array, pxu.EnergyMinimizationLoop.LoopState]:
    logging.info(f"Jitting train_on_batch for model {id(model)}")
    t_iterations = model.p.T
    if model.p.pc_mode == "efficient_ppc":
        t_iterations -= model.p.T_min_w_updates
    total_iterations = model.p.T
    if model.p.pc_mode == "pc":
        total_iterations += 1

    with pxu.train(model, examples):
        t_loop = pxu.EnergyMinimizationLoop(
            model=model,
            loss_fn=loss_fn,
            max_iter_number=t_iterations,
            min_iter_number=model.p.T_min_x_updates,
            energy_convergence_threshold=(
                model.p.energy_quick_approximate_convergence_threshold
                if model.p.pc_mode == "efficient_ppc"
                else None
            ),
            should_update_x=True,
            should_update_w=model.p.pc_mode == "ppc",
            optim_x=optim_x,
            optim_w=optim_w if model.p.pc_mode == "ppc" else None,
        )
        t_loop_state = t_loop.run(examples, recording_buffer_length=total_iterations)
        final_state = t_loop_state

        if model.p.pc_mode in ["pc", "efficient_ppc"]:
            min_iter_number = (
                model.p.T_min_w_updates if model.p.pc_mode == "efficient_ppc" else 1
            )
            w_loop = pxu.EnergyMinimizationLoop(
                model=model,
                loss_fn=loss_fn,
                max_iter_number=total_iterations,
                min_iter_number=min_iter_number + t_loop_state.iter_number,
                energy_convergence_threshold=(
                    model.p.energy_slow_accurate_convergence_threshold
                    if model.p.pc_mode == "efficient_ppc"
                    else None
                ),
                should_update_x=model.p.pc_mode == "efficient_ppc",
                should_update_w=True,
                optim_x=optim_x if model.p.pc_mode == "efficient_ppc" else None,
                optim_w=optim_w,
            )
            w_loop_state = w_loop.run(examples, initial_state=t_loop_state)
            final_state = w_loop_state

    predictions = feed_forward_predict(model.internal_state, model=model)[0]
    mse = jnp.mean((predictions - examples) ** 2)
    return (mse, final_state)


@pxu.jit()
def test_on_batch(
    examples: jax.Array,
    *,
    model: PCDecoder,
    optim_x: pxu.Optim,
    loss_fn: Callable,
) -> tuple[jax.Array, pxu.EnergyMinimizationLoop.LoopState]:
    logging.info(f"Jitting test_on_batch for model {id(model)}")
    final_state = model.converge_on_batch(examples, optim_x=optim_x, loss_fn=loss_fn)
    predictions = feed_forward_predict(model.internal_state, model=model)[0]
    mse = jnp.mean((predictions - examples) ** 2)
    return (mse, final_state)


@pxu.jit()
def get_internal_states_on_batch(
    examples, *, model: PCDecoder, optim_x, loss_fn
) -> jax.Array:
    logging.info(f"Jitting get_internal_states_on_batch for model {id(model)}")
    model.converge_on_batch(examples, optim_x=optim_x, loss_fn=loss_fn)
    assert model.internal_state is not None
    return model.internal_state


def run_training_experiment(params: Params) -> None:
    results_dir = Path(params.results_dir) / params.experiment_name  # type: ignore
    if results_dir.exists() and any(results_dir.iterdir()):
        if params.do_hypertunning and params.hypertunning_resume_run:
            shutil.move(
                results_dir, results_dir.with_suffix(f".backup-{tune.get_trial_id()}")
            )
        elif params.overwrite_results_dir:
            shutil.rmtree(results_dir)
        else:
            raise RuntimeError(
                f"Results dir {results_dir} already exists and is not empty!"
            )
    results_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(params)

    if params.load_weights_from is not None:
        model.load_weights(params.load_weights_from)

    if params.wandb_logging:
        with init_wandb(params=params, results_dir=results_dir) as run:
            train_model(model=model, params=params, results_dir=results_dir, run=run)
    else:
        train_model(model=model, params=params, results_dir=results_dir)


def build_model(params: Params) -> PCDecoder:
    return PCDecoder(
        params=params,
        internal_state_init_fn=internal_state_init,
    )


def build_optim_x(model: PCDecoder, params: Params) -> pxu.Optim:
    if params.optimizer_x == "sgd":
        optim_x = pxu.Optim(
            optax.chain(
                optax.add_decayed_weights(weight_decay=params.optim_x_l2),
                optax.sgd(params.optim_x_lr / params.batch_size),
            ),
            model.x_parameters(),
            allow_none_grads=True,
        )
    elif params.optimizer_x == "adamw":
        optim_x = pxu.Optim(
            optax.adamw(
                params.optim_x_lr / params.batch_size,
                weight_decay=params.optim_x_l2,
            ),
            model.x_parameters(),
            allow_none_grads=True,
        )
    else:
        raise ValueError(f"Unknown optimizer_x: {params.optimizer_x}")
    return optim_x


def build_optim_w(model: PCDecoder, params: Params) -> pxu.Optim:
    if params.optimizer_w == "sgd":
        optim_w = pxu.Optim(
            optax.chain(
                optax.add_decayed_weights(weight_decay=params.optim_w_l2),
                optax.sgd(params.optim_w_lr / params.batch_size),
            ),
            model.w_parameters(),
            allow_none_grads=True,
        )
    elif params.optimizer_w == "adamw":
        optim_w = pxu.Optim(
            optax.chain(
                optax.adamw(
                    params.optim_w_lr / params.batch_size,
                    weight_decay=params.optim_w_l2,
                ),
            ),
            model.w_parameters(),
            allow_none_grads=True,
        )
    else:
        raise ValueError(f"Unknown optimizer_w: {params.optimizer_w}")
    return optim_w


def train_model(
    model: PCDecoder,
    params: Params,
    results_dir: Path,
    run: wandb.wandb_sdk.wandb_run.Run | None = None,
) -> None:
    best_epoch_dir = results_dir / "best"
    with pxu.train(model, jnp.zeros((params.batch_size, params.output_dim))):
        optim_x = build_optim_x(model=model, params=params)
        optim_w = build_optim_w(model=model, params=params)

    train_batch_fn = train_on_batch.snapshot(
        model=model,
        optim_x=optim_x,
        optim_w=optim_w,
        loss_fn=model_energy_loss,  # type: ignore
    )

    test_batch_fn = test_on_batch.snapshot(
        model=model,
        optim_x=optim_x,
        loss_fn=model_energy_loss,  # type: ignore
    )

    train_loader, test_loader, train_data_mean, train_data_std = get_data_loaders(
        params
    )
    if DEBUG:
        train_loader = ReentryIsliceIterator(train_loader, DEBUG_BATCH_NUMBER)  # type: ignore
        test_loader = ReentryIsliceIterator(test_loader, DEBUG_BATCH_NUMBER)  # type: ignore

    train_mses = []
    test_mses = []
    best_train_mse = float("inf")
    best_test_mse = float("inf")
    train_t_step: int = 0
    test_t_step: int = 0

    with tqdm(range(params.epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Train Epoch {epoch + 1}")
            logging.info(f"Starting epoch {epoch + 1}")

            epoch_train_mses = []
            with tqdm(train_loader, unit="batch") as tbatch:
                for examples, _ in tbatch:
                    tbatch.set_description(f"Train Batch {tbatch.n + 1}")
                    if params.reset_optimizer_x_state:
                        optim_x.init_state()
                    mse, final_state = train_batch_fn(examples)
                    mse = mse.item()
                    all_energies = final_state.all_energies[
                        : final_state.iter_number.item()
                    ]
                    log_train_t_step_metrics(
                        run=run,
                        t_step=train_t_step,
                        energies=all_energies,
                    )
                    log_train_batch_metrics(
                        run=run,
                        epochs=epoch,
                        batches_per_epoch=len(train_loader),
                        batch=tbatch.n,
                        num_x_updates=final_state.num_x_updates_done.item(),
                        num_w_updates=final_state.num_w_updates_done.item(),
                    )
                    train_t_step += final_state.iter_number.item()
                    epoch_train_mses.append(mse)
                    tbatch.set_postfix(mse=mse)

                    # Force GC to free some RAM and GPU memory
                    del final_state
                    gc.collect()

            epoch_train_mse: float = float(
                np.mean(
                    epoch_train_mses[-params.use_last_n_batches_to_compute_metrics :]
                )
            )
            train_mses.append(epoch_train_mse)
            logging.info(f"Finished training in epoch {epoch + 1}")

            epoch_test_mses = []
            with tqdm(test_loader, unit="batch") as tbatch:
                for examples, _ in tbatch:
                    tbatch.set_description(f"Test Batch {tbatch.n + 1}")
                    if params.reset_optimizer_x_state:
                        optim_x.init_state()
                    mse, final_state = test_batch_fn(examples)
                    mse = mse.item()
                    all_energies = final_state.all_energies[
                        : final_state.iter_number.item()
                    ]
                    log_test_t_step_metrics(
                        run=run,
                        t_step=test_t_step,
                        energies=all_energies,
                    )
                    test_t_step += final_state.iter_number.item()
                    epoch_test_mses.append(mse)
                    tbatch.set_postfix(mse=mse)
            epoch_test_mse: float = float(np.mean(epoch_test_mses))
            test_mses.append(epoch_test_mse)
            logging.info(f"Finished testing in epoch {epoch + 1}")

            epoch_report = {
                "epochs": epoch + 1,
                "train_mse": epoch_train_mse,
                "test_mse": epoch_test_mse,
            }

            should_save_intermediate_results = (
                params.save_intermediate_results
                and (epoch + 1) % params.save_results_every_n_epochs == 0
            )
            should_save_best_results = (
                params.save_best_results and epoch_test_mse < best_test_mse
            )
            best_train_mse = min(best_train_mse, epoch_train_mse)
            best_test_mse = min(best_test_mse, epoch_test_mse)
            if should_save_intermediate_results or should_save_best_results:
                logging.info(
                    f"Saving results for epoch {epoch + 1}. Best epoch: {should_save_best_results}. MSE: {epoch_test_mse}"
                )
                epoch_results = results_dir / f"epochs_{epoch + 1}"
                epoch_results.mkdir()

                if should_save_best_results:
                    best_epoch_dir.unlink(missing_ok=True)
                    best_epoch_dir.symlink_to(
                        epoch_results.relative_to(results_dir),
                        target_is_directory=True,
                    )

                model.save_weights(str(epoch_results))  # type: ignore
                with open(os.path.join(epoch_results, "report.json"), "w") as outfile:
                    json.dump(
                        dict(**epoch_report, params=params.dict()), outfile, indent=4
                    )

            if run is not None:
                run.log(epoch_report)
            if params.do_hypertunning:
                session.report(epoch_report)

            logging.info(f"Finished epoch {epoch + 1}")
            tepoch.set_postfix(train_mse=epoch_train_mse, test_mse=epoch_test_mse)

    # Generate images for the best epoch only
    if best_epoch_dir.exists():
        visualize_epoch(
            epoch_dir=best_epoch_dir,
            run=run,
            test_loader=test_loader,
            train_data_mean=train_data_mean,
            train_data_std=train_data_std,
        )
    else:
        logging.error(f"No best epoch exists for this run")

    if run is not None:
        run.summary["train_mse"] = best_train_mse
        run.summary["test_mse"] = best_test_mse

    logging.info(
        f"Finished training for {params.epochs} epochs, test_mse={best_test_mse}"
    )


class EpochResults(NamedTuple):
    epochs: int
    epoch_report: dict
    epoch_dir: Path
    params: Params
    model: PCDecoder
    optim_x: pxu.Optim
    optim_w: pxu.Optim


def load_epoch(
    epoch_dir: Path,
) -> EpochResults:
    if not epoch_dir.exists():
        raise ValueError(f"Epoch dir {epoch_dir} does not exist!")
    logging.info(f"Loading epoch from {epoch_dir.resolve()}...")
    with open(epoch_dir / "report.json") as infile:
        epoch_report = json.load(infile)
        epochs = epoch_report["epochs"]
    params = Params(**epoch_report["params"])
    model = build_model(params)
    model.load_weights(str(epoch_dir))
    with pxu.train(model, jnp.zeros((params.batch_size, params.output_dim))):
        optim_x = build_optim_x(model=model, params=params)
        optim_w = build_optim_w(model=model, params=params)
    logging.info(f"Loaded epoch from {epoch_dir.resolve()}.")
    return EpochResults(
        epochs=epochs,
        epoch_report=epoch_report,
        epoch_dir=epoch_dir,
        params=params,
        model=model,
        optim_x=optim_x,
        optim_w=optim_w,
    )


def visualize_epoch(
    *,
    epoch_dir: Path,
    run: wandb.wandb_sdk.wandb_run.Run | None,
    test_loader,
    train_data_mean: float,
    train_data_std: float,
) -> None:
    epoch_results = load_epoch(epoch_dir=epoch_dir)
    params = epoch_results.params
    model = epoch_results.model
    optim_x = epoch_results.optim_x
    if params.reset_optimizer_x_state:
        # It's not needed right now, because the optimizer state is not saved and thus not loaded.
        # However, let's keep it here in case we decide to save the optimizer state in the future.
        optim_x.init_state()
    examples, labels = get_stratified_test_batch(test_loader)
    internal_states = get_internal_states_on_batch(
        examples=examples,
        model=model,
        optim_x=optim_x,
        loss_fn=model_energy_loss,
    )
    predictions = feed_forward_predict(internal_states, model=epoch_results.model)[0]
    create_all_visualizations(
        out_dir=epoch_dir,
        run=run,
        epochs=epoch_results.epochs,
        examples=examples,
        labels=labels,
        internal_states=internal_states,
        predictions=predictions,
        params=epoch_results.params,
        train_data_mean=train_data_mean,
        train_data_std=train_data_std,
    )
    internal_states_mean = jnp.mean(internal_states).item()
    internal_states_std = jnp.std(internal_states).item()
    with open(epoch_dir / "report.json", "w") as outfile:
        json.dump(
            {
                **epoch_results.epoch_report,
                "internal_states_mean": internal_states_mean,
                "internal_states_std": internal_states_std,
            },
            outfile,
            indent=4,
        )
    if run is not None:
        run.summary["internal_states_mean"] = internal_states_mean
        run.summary["internal_states_std"] = internal_states_std
    logging.info(f"Finished visualizing epoch from {epoch_dir}...")


class Trainable:
    def __init__(self, params: Params) -> None:
        self.params = params

    def __call__(self, config: dict):
        gc.collect()
        params = self.params.update(config, inplace=False, validate=True)
        # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.utils.wait_for_gpu.html#ray.tune.utils.wait_for_gpu
        if params.hypertunning_gpu_memory_fraction_per_trial > 0:
            tune.utils.wait_for_gpu(  # type: ignore
                target_util=1.0 - params.hypertunning_gpu_memory_fraction_per_trial,
                retry=50,
                delay_s=12,
            )
        run_training_experiment(params)
        gc.collect()
