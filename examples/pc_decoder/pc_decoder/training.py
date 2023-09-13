import gc
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
import wandb
from pc_decoder.data_loading import get_data_loaders, get_stratified_test_batch
from pc_decoder.logging import init_wandb, log_batch_metrics, log_t_step_metrics
from pc_decoder.model import PCDecoder, feed_forward_predict, model_energy_loss
from pc_decoder.params import Params
from pc_decoder.visualization import create_all_visualizations
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

import pcax.utils as pxu  # type: ignore

DEBUG = os.environ.get("DEBUG", "0") == "1"
# DEBUG_TRAIN_SUBSET_INDICES = [1]
# DEBUG_TEST_SUBSET_INDICES = [3]
DEBUG_TRAIN_SUBSET_INDICES = list(range(10))
DEBUG_TEST_SUBSET_INDICES = DEBUG_TRAIN_SUBSET_INDICES
assert len(DEBUG_TRAIN_SUBSET_INDICES) == len(DEBUG_TEST_SUBSET_INDICES)


def internal_state_init(
    params: Params,
    prng_key: jax.random.KeyArray,
) -> tuple[jax.Array, jax.random.KeyArray]:
    # TODO: Play with different initialization strategies
    value = jnp.zeros((params.internal_dim,))
    return value, prng_key


def build_model(params: Params) -> PCDecoder:
    return PCDecoder(
        params=params,
        internal_state_init_fn=internal_state_init,
    )


# We can try gradient clipping in optimizers.


def build_optim_x(model: PCDecoder, params: Params) -> pxu.Optim:
    # Do not divide X learning rate by batch size, as we have a dedicated set of X parameters for each example in a batch.
    # Each X parameter actually has a batch dimension, even though vmap makes it look otherwise inside the model code.
    # Thus, X's gradients also have batch dimensions, and gradients from different examples are never summed together.
    learning_rate = params.optimizer_x_learning_rate
    if params.optimizer_x == "sgd":
        optim_x = pxu.Optim(
            optax.chain(
                optax.add_decayed_weights(weight_decay=params.optimizer_x_weight_decay),
                optax.sgd(
                    learning_rate=learning_rate,
                    momentum=params.optimizer_x_sgd_momentum,
                    nesterov=True,
                ),
            ),
            model.x_parameters(),
            allow_none_grads=True,
        )
    elif params.optimizer_x == "adamw":
        optim_x = pxu.Optim(
            optax.adamw(
                learning_rate=learning_rate,
                b1=params.optimizer_x_adamw_beta1,
                b2=params.optimizer_x_adamw_beta2,
                weight_decay=params.optimizer_x_weight_decay,
            ),
            model.x_parameters(),
            allow_none_grads=True,
        )
    else:
        raise ValueError(f"Unknown optimizer_x: {params.optimizer_x}")
    return optim_x


def build_optim_w(model: PCDecoder, params: Params) -> pxu.Optim:
    # Divide W learning rate by batch size, as we have a single set of W parameters for all examples in a batch.
    # The resuting energy is summed across all examples in the batch, see the @pxu.vectorize decorator on model_energy_loss.
    # Thus, the magnitude of gradient for each W param is proprtional to the batch size,
    # and we need to divide the learning rate by the batch size to get the average gradient per batch.
    learning_rate = params.optimizer_w_learning_rate / params.batch_size
    if params.optimizer_w == "sgd":
        optim_w = pxu.Optim(
            optax.chain(
                optax.add_decayed_weights(weight_decay=params.optimizer_w_weight_decay),
                optax.sgd(
                    learning_rate=learning_rate,
                    momentum=params.optimizer_w_sgd_momentum,
                    nesterov=True,
                ),
            ),
            model.w_parameters(),
            allow_none_grads=True,
        )
    elif params.optimizer_w == "adamw":
        optim_w = pxu.Optim(
            optax.chain(
                optax.adamw(
                    learning_rate=learning_rate,
                    b1=params.optimizer_w_adamw_beta1,
                    b2=params.optimizer_w_adamw_beta2,
                    weight_decay=params.optimizer_w_weight_decay,
                ),
            ),
            model.w_parameters(),
            allow_none_grads=True,
        )
    else:
        raise ValueError(f"Unknown optimizer_w: {params.optimizer_w}")
    return optim_w


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


class EpochResults(NamedTuple):
    epoch_dir: Path
    params: Params
    epoch: int
    train_mse: float
    test_mse: float
    model: PCDecoder
    optim_x: pxu.Optim
    optim_w: pxu.Optim

    @property
    def report(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_mse": self.train_mse,
            "test_mse": self.test_mse,
        }

    @property
    def has_nans(self) -> bool:
        return np.isnan(self.train_mse) or np.isnan(self.test_mse)

    def save(self) -> None:
        logging.info(f"Saving epoch {self.epoch} to {self.epoch_dir.resolve()} ...")
        self.epoch_dir.mkdir(parents=True, exist_ok=False)
        with open(self.epoch_dir / "params.json", "w") as outfile:
            json.dump(self.params.dict(), outfile, indent=4)
        with open(self.epoch_dir / "report.json", "w") as outfile:
            json.dump(self.report, outfile, indent=4)
        self.model.save_weights(str(self.epoch_dir))  # type: ignore
        logging.info(f"Saved epoch {self.epoch} to {self.epoch_dir.resolve()} .")

    @classmethod
    def load(cls, epoch_dir: Path) -> "EpochResults":
        if not epoch_dir.exists():
            raise ValueError(f"Epoch dir {epoch_dir} does not exist!")
        logging.info(f"Loading epoch from {epoch_dir.resolve()} ...")
        with open(epoch_dir / "params.json") as infile:
            params_dict = json.load(infile)
            params = Params(**params_dict)
        with open(epoch_dir / "report.json") as infile:
            report = json.load(infile)
        model = build_model(params)
        model.load_weights(str(epoch_dir))
        with pxu.train(model, jnp.zeros((params.batch_size, params.output_dim))):
            optim_x = build_optim_x(model=model, params=params)
            optim_w = build_optim_w(model=model, params=params)
        epoch_result = EpochResults(
            epoch_dir=epoch_dir,
            params=params,
            epoch=report["epoch"],
            train_mse=report["train_mse"],
            test_mse=report["test_mse"],
            model=model,
            optim_x=optim_x,
            optim_w=optim_w,
        )
        logging.info(f"Loaded epoch {epoch_result.epoch} from {epoch_dir.resolve()}.")
        return epoch_result


def visualize_epoch(
    *,
    epoch_results: EpochResults,
    run: wandb.wandb_sdk.wandb_run.Run | None,
    test_loader: DataLoader[Any],
    train_data_mean: float,
    train_data_std: float,
) -> None:
    if epoch_results.has_nans:
        logging.error(f"Epoch {epoch_results.epoch} has NaNs, skipping visualization")
        return
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
        out_dir=epoch_results.epoch_dir,
        run=run,
        epoch=epoch_results.epoch,
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
    with open(epoch_results.epoch_dir / "internal_states.json", "w") as outfile:
        json.dump(
            {
                "internal_states_mean": internal_states_mean,
                "internal_states_std": internal_states_std,
                "internal_states": internal_states.tolist(),
                "labels": labels.tolist(),
            },
            outfile,
            indent=4,
        )
    if run is not None:
        run.summary["internal_states_mean"] = internal_states_mean
        run.summary["internal_states_std"] = internal_states_std
    logging.info(
        f"Finished visualizing epoch {epoch_results.epoch} from {epoch_results.epoch_dir} ..."
    )


class TrainingRun:
    params: Params

    results_dir: Path
    best_epoch_dir: Path
    name: str
    run: wandb.wandb_sdk.wandb_run.Run | None

    model: PCDecoder
    optim_x: pxu.Optim
    optim_w: pxu.Optim

    train_loader: DataLoader[Any]
    test_loader: DataLoader[Any]
    train_data_mean: float
    train_data_std: float

    train_batch_fn: Callable
    test_batch_fn: Callable

    epochs_done: int = 0
    train_t_step: int = 0
    test_t_step: int = 0

    train_mses: list[float]
    test_mses: list[float]
    best_train_mse: float = float("inf")
    best_test_mse: float = float("inf")

    def __init__(
        self,
        params: Params,
        *,
        results_dir: str | None = None,
        name: str | None = None,
        model: PCDecoder | None = None,
    ):
        self.params = params.copy()
        if DEBUG:
            self.params.batch_size = min(
                self.params.batch_size, len(DEBUG_TRAIN_SUBSET_INDICES)
            )

        if name is None:
            run_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.name = f"{self.params.experiment_name}--{run_time}"
        else:
            self.name = name

        if results_dir is None:
            self.results_dir: Path = (
                Path(self.params.results_dir) / self.params.experiment_name
            )
            if self.results_dir.exists() and any(self.results_dir.iterdir()):
                if self.params.do_hypertunning and self.params.hypertunning_resume_run:
                    shutil.move(
                        self.results_dir,
                        self.results_dir.with_suffix(f".backup-{self.name}"),
                    )
                elif self.params.overwrite_results_dir:
                    shutil.rmtree(self.results_dir)
                else:
                    raise RuntimeError(
                        f"Results dir {self.results_dir} already exists and it is not empty!"
                    )
            self.results_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.results_dir = Path(results_dir)
            if not self.results_dir.exists():
                raise ValueError(f"Results dir {self.results_dir} does not exist!")
        self.best_epoch_dir = self.results_dir / "best"

        self.run = None
        if self.params.wandb_logging:
            self.run = init_wandb(
                run_name=self.name, params=self.params, results_dir=self.results_dir
            )

        if model is None:
            self.model = build_model(self.params)
            if self.params.load_weights_from is not None:
                self.model.load_weights(self.params.load_weights_from)
        else:
            self.model = model

        with pxu.train(
            self.model, jnp.zeros((self.params.batch_size, self.params.output_dim))
        ):
            self.optim_x = build_optim_x(model=self.model, params=self.params)
            self.optim_w = build_optim_w(model=self.model, params=self.params)

        (
            self.train_loader,
            self.test_loader,
            self.train_data_mean,
            self.train_data_std,
        ) = get_data_loaders(
            self.params,
            train_subset_indices=DEBUG_TRAIN_SUBSET_INDICES if DEBUG else None,
            test_subset_indices=DEBUG_TEST_SUBSET_INDICES if DEBUG else None,
        )

        self.train_batch_fn = train_on_batch.snapshot(
            model=self.model,
            optim_x=self.optim_x,
            optim_w=self.optim_w,
            loss_fn=model_energy_loss,  # type: ignore
        )

        self.test_batch_fn = test_on_batch.snapshot(
            model=self.model,
            optim_x=self.optim_x,
            loss_fn=model_energy_loss,  # type: ignore
        )

        self.train_mses = []
        self.test_mses = []

    def _process_all_batches(self, *, mode: str) -> list[float]:
        assert mode in ["train", "test"]
        data_loader = self.train_loader if mode == "train" else self.test_loader
        process_batch_fn = (
            self.train_batch_fn if mode == "train" else self.test_batch_fn
        )
        t_step_attr = "train_t_step" if mode == "train" else "test_t_step"

        epoch_mses = []
        with tqdm(data_loader, unit="batch") as tbatch:
            for examples, _ in tbatch:
                batch = tbatch.n + 1
                tbatch.set_description(f"{mode.capitalize()} Batch {batch}")
                if self.params.reset_optimizer_x_state:
                    self.optim_x.init_state()
                mse, final_state = process_batch_fn(examples)
                mse = mse.item()
                all_energies = final_state.all_energies[
                    : final_state.iter_number.item()
                ]
                if self.params.log_t_metrics:
                    log_t_step_metrics(
                        mode=mode,
                        run=self.run,
                        t_step=getattr(self, t_step_attr),
                        energies=all_energies,
                    )
                log_batch_metrics(
                    mode=mode,
                    run=self.run,
                    epochs_done=self.epochs_done,
                    batches_per_epoch=len(data_loader),
                    batch=batch,
                    final_energies=all_energies[-1],
                    num_x_updates=final_state.num_x_updates_done.item(),
                    num_w_updates=final_state.num_w_updates_done.item(),
                )
                setattr(
                    self,
                    t_step_attr,
                    getattr(self, t_step_attr) + final_state.iter_number.item(),
                )
                epoch_mses.append(mse)
                tbatch.set_postfix(**{f"batch_{mode}_mse": f"{mse:.4f}"})

                # Force GC to free some RAM and GPU memory
                del final_state
                gc.collect()
        return epoch_mses

    def train_for_epoch(self) -> EpochResults:
        logging.info(f"Starting epoch {self.epochs_done + 1}")

        epoch_train_mses = self._process_all_batches(mode="train")
        epoch_train_mse: float = float(
            np.mean(
                epoch_train_mses[-self.params.use_last_n_batches_to_compute_metrics :]
            )
        )
        self.train_mses.append(epoch_train_mse)
        logging.info(f"Finished training epoch {self.epochs_done + 1}")

        epoch_test_mses = self._process_all_batches(mode="test")
        epoch_test_mse: float = float(np.mean(epoch_test_mses))
        self.test_mses.append(epoch_test_mse)
        logging.info(f"Finished testing epoch {self.epochs_done + 1}")

        # self._process_all_batches relies on self.epochs_done to not include the current epoch.
        self.epochs_done += 1

        epoch_results = EpochResults(
            epoch_dir=self.results_dir / f"epochs_{self.epochs_done}",
            params=self.params,
            epoch=self.epochs_done,
            train_mse=epoch_train_mse,
            test_mse=epoch_test_mse,
            model=self.model,
            optim_x=self.optim_x,
            optim_w=self.optim_w,
        )

        should_save_intermediate_results = (
            self.params.save_intermediate_results
            and (self.epochs_done) % self.params.save_results_every_n_epochs == 0
        )
        should_save_best_results = (
            self.params.save_best_results and epoch_test_mse < self.best_test_mse
        )
        self.best_train_mse = min(self.best_train_mse, epoch_train_mse)
        self.best_test_mse = min(self.best_test_mse, epoch_test_mse)
        if should_save_intermediate_results or should_save_best_results:
            logging.info(
                f"Saving results for epoch {self.epochs_done}. Best epoch: {should_save_best_results}. MSE: {epoch_test_mse}"
            )
            epoch_results.save()

            if should_save_best_results:
                self.best_epoch_dir.unlink(missing_ok=True)
                self.best_epoch_dir.symlink_to(
                    epoch_results.epoch_dir.relative_to(self.results_dir),
                    target_is_directory=True,
                )

        if self.run is not None:
            self.run.log(epoch_results.report)

        logging.info(f"Finished epoch {self.epochs_done}")

        return epoch_results

    def finish(self, status: str | None = None) -> None:
        if self.epochs_done == 0:
            logging.warning(
                f"Attempted to finish run {self.name} that completed no epochs."
            )

        best_epoch_results: EpochResults | None = None
        # Generate images for the best epoch only
        if self.best_epoch_dir.exists():
            best_epoch_results = EpochResults.load(self.best_epoch_dir)
            visualize_epoch(
                epoch_results=best_epoch_results,
                run=self.run,
                test_loader=self.test_loader,
                train_data_mean=self.train_data_mean,
                train_data_std=self.train_data_std,
            )
        else:
            logging.error(
                f"No best epoch exists for run {self.name}. The path {self.best_epoch_dir.resolve()} was not found."
            )

        if status is None:
            if self.epochs_done < self.params.epochs:
                status = "interrupted"
            elif self.epochs_done == self.params.epochs:
                status = "completed"

        if self.run is not None:
            self.run.summary["status"] = status
            self.run.summary["epoch"] = self.epochs_done
            if best_epoch_results is not None:
                self.run.summary["best_epoch"] = best_epoch_results.epoch
            self.run.summary["train_mse"] = self.best_train_mse
            self.run.summary["test_mse"] = self.best_test_mse

        logging.warning(
            f"<<<<<<<<<<---------- Run {self.name} finished training with status '{status}' after {self.epochs_done} epochs, test_mse={self.best_test_mse} ---------->>>>>>>>>>"
        )

        if self.run is not None:
            self.run.finish()

    def save(self, savedir: str) -> None:
        state = {
            "params": self.params.dict(),
            "results_dir": str(self.results_dir),
            "name": self.name,
            "epochs_done": self.epochs_done,
            "train_t_step": self.train_t_step,
            "test_t_step": self.test_t_step,
            "train_mses": self.train_mses,
            "test_mses": self.test_mses,
            "best_train_mse": self.best_train_mse,
            "best_test_mse": self.best_test_mse,
        }
        with open(os.path.join(savedir, "training_run_state.json"), "w") as outfile:
            json.dump(state, outfile, indent=4)
        self.model.save_weights(savedir)

    @classmethod
    def load(cls, savedir: str) -> "TrainingRun":
        with open(os.path.join(savedir, "training_run_state.json")) as infile:
            state = json.load(infile)
        params = Params(**state["params"])
        model = build_model(params)
        model.load_weights(savedir)
        run = cls(
            params, results_dir=state["results_dir"], name=state["name"], model=model
        )
        run.epochs_done = state["epochs_done"]
        run.train_t_step = state["train_t_step"]
        run.test_t_step = state["test_t_step"]
        run.train_mses = state["train_mses"]
        run.test_mses = state["test_mses"]
        run.best_train_mse = state["best_train_mse"]
        run.best_test_mse = state["best_test_mse"]
        return run


class ExperimentStopper(Protocol):
    def should_stop(self, metrics: dict) -> bool:
        ...

    def stop_reason(self, metrics: dict) -> str:
        ...

    def set_experiment_name(self, name: str) -> None:
        ...


class ExperimentStopperComposite:
    def __init__(self, stoppers: list[ExperimentStopper]) -> None:
        self.stoppers = stoppers
        self._stop_reason: str = ""

    def should_stop(self, metrics: dict) -> bool:
        self._stop_reason = ""
        for stopper in self.stoppers:
            if stopper.should_stop(metrics):
                self._stop_reason = stopper.stop_reason(metrics)
                return True
        return False

    def stop_reason(self, metrics: dict) -> str:
        return self._stop_reason

    def set_experiment_name(self, name: str) -> None:
        for stopper in self.stoppers:
            stopper.set_experiment_name(name)


def run_training_experiment(
    *,
    params: Params,
    stopper: ExperimentStopper | None = None,
) -> None:
    training_run = TrainingRun(params=params)
    if stopper is not None:
        stopper.set_experiment_name(training_run.name)

    finish_status = None
    try:
        with tqdm(range(params.epochs), unit="epoch") as tepoch:
            for epoch_index in tepoch:
                epoch = epoch_index + 1
                tepoch.set_description(f"Train Epoch {epoch}")

                result = training_run.train_for_epoch()

                tepoch.set_postfix(train_mse=result.train_mse, test_mse=result.test_mse)
                if stopper is not None and stopper.should_stop(result.report):
                    finish_status = stopper.stop_reason(result.report)
                    logging.warning(
                        f"Experiment stopper decided to stop the experiment {training_run.name} after {epoch} epochs"
                    )
                    break
    except Exception as exc:
        finish_status = "failed"
        raise
    finally:
        training_run.finish(status=finish_status)
