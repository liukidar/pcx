from datetime import datetime
from pathlib import Path

import jax
import wandb
from pc_decoder.params import Params
from ray import tune


def init_wandb(
    *,
    params: Params,
    results_dir: Path,
) -> wandb.wandb_sdk.wandb_run.Run:
    run_name = (
        f"{params.experiment_name}--{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    if params.do_hypertunning:
        run_name += f"--{tune.get_trial_id()}"  # type: ignore
    params_update = {
        "results_dir": str(results_dir.absolute()),  # type: ignore
    }
    if params.do_hypertunning:
        params_update["trial_id"] = tune.get_trial_id()  # type: ignore
        params_update["trial_name"] = tune.get_trial_name()  # type: ignore
        params_update["trial_dir"] = tune.get_trial_dir()  # type: ignore
    run = wandb.init(
        project="pc-decoder",
        group=params.experiment_name,
        name=run_name,
        tags=["predictive-coding", "autoencoder", "decoder"],
        config={
            **params.dict(),
            **params_update,
        },
    )
    wandb.define_metric("train_energy/*", step_metric="train_t_step")
    wandb.define_metric("test_energy/*", step_metric="test_t_step")
    wandb.define_metric("train_mse", step_metric="epochs")
    wandb.define_metric("test_mse", step_metric="epochs")
    wandb.define_metric("internal_states_mean", step_metric="epochs")
    wandb.define_metric("internal_states_std", step_metric="epochs")
    wandb.define_metric("num_x_updates", step_metric="batch")
    wandb.define_metric("num_w_updates", step_metric="batch")
    assert isinstance(run, wandb.wandb_sdk.wandb_run.Run)
    return run


def log_train_t_step_metrics(
    *,
    run: wandb.wandb_sdk.wandb_run.Run | None,
    t_step: int,
    energies: jax.Array,
) -> None:
    if run is None:
        return
    total_energies = energies.sum(axis=1).tolist()
    for t, t_energies in enumerate(energies):
        t_metrics = {
            "train_t_step": t_step + t,
            "train_energy/0_total": total_energies[t],
        }
        for node_index, energy in enumerate(t_energies):
            t_metrics[
                f"train_energy/{len(t_energies) - node_index}_node_{node_index}"
            ] = energy.item()

        run.log(t_metrics)


def log_train_batch_metrics(
    *,
    run: wandb.wandb_sdk.wandb_run.Run | None,
    epochs: int,
    batches_per_epoch: int,
    batch: int,
    num_x_updates: int,
    num_w_updates: int,
) -> None:
    if run is None:
        return
    run.log(
        {
            "batch": epochs * batches_per_epoch + batch,
            "num_x_updates": num_x_updates,
            "num_w_updates": num_w_updates,
        }
    )


def log_test_t_step_metrics(
    *,
    run: wandb.wandb_sdk.wandb_run.Run | None,
    t_step: int,
    energies: jax.Array,
) -> None:
    if run is None:
        return
    total_energy = energies.sum(axis=1).tolist()
    for t, t_energies in enumerate(energies):
        t_metrics = {
            "test_t_step": t_step + t,
            "test_energy/0_total": total_energy[t],
        }
        for node_index, energy in enumerate(t_energies):
            t_metrics[
                f"test_energy/{len(t_energies) - node_index}_node_{node_index}"
            ] = energy.item()

        run.log(t_metrics)
