from datetime import datetime
from pathlib import Path

import jax
import wandb
from pc_decoder.params import Params


def init_wandb(
    *,
    run_name: str,
    params: Params,
    results_dir: Path,
) -> wandb.wandb_sdk.wandb_run.Run:
    run = wandb.init(
        project="pc-decoder",
        group=params.experiment_name,
        name=run_name,
        tags=["predictive-coding", "autoencoder", "decoder"],
        config={
            **params.dict(),
            "results_dir": str(results_dir.absolute()),  # type: ignore
        },
    )
    assert isinstance(run, wandb.wandb_sdk.wandb_run.Run)
    for step_metric in [
        "train/t_step",
        "test/t_step",
        "train/batch",
        "test/batch",
        "epoch",
    ]:
        wandb.define_metric(step_metric, hidden=True)
    wandb.define_metric("train/t/*", step_metric="train/t_step")
    wandb.define_metric("test/t/*", step_metric="test/t_step")
    wandb.define_metric("train/batch/*", step_metric="train/batch")
    wandb.define_metric("test/batch/*", step_metric="test/batch")
    wandb.define_metric("train_mse", step_metric="epoch", goal="minimize")
    wandb.define_metric("test_mse", step_metric="epoch", goal="minimize")
    return run


def log_t_step_metrics(
    *,
    mode: str,
    run: wandb.wandb_sdk.wandb_run.Run | None,
    t_step: int,
    energies: jax.Array,
) -> None:
    assert mode in ["train", "test"]
    if run is None:
        return
    total_energies = energies.sum(axis=1).tolist()
    for t, t_energies in enumerate(energies):
        t_metrics = {
            f"{mode}/t_step": t_step + t + 1,
            f"{mode}/t/energy/0_total": total_energies[t],
        }
        for node_index, energy in enumerate(t_energies):
            t_metrics[
                f"{mode}/t/energy/{len(t_energies) - node_index}_node_{node_index}"
            ] = energy.item()

        run.log(t_metrics)


def log_batch_metrics(
    *,
    mode: str,
    run: wandb.wandb_sdk.wandb_run.Run | None,
    epochs_done: int,
    batches_per_epoch: int,
    batch: int,
    final_energies: jax.Array,
    num_x_updates: int,
    num_w_updates: int,
) -> None:
    assert mode in ["train", "test"]
    if run is None:
        return
    metrics = {
        f"{mode}/batch": epochs_done * batches_per_epoch + batch,
        f"{mode}/batch/energy/0_total": final_energies.sum(),
        f"{mode}/batch/num_x_updates": num_x_updates,
        f"{mode}/batch/num_w_updates": num_w_updates,
    }
    for node_index, energy in enumerate(final_energies):
        metrics[
            f"{mode}/batch/energy/{len(final_energies) - node_index}_node_{node_index}"
        ] = energy.item()
    run.log(metrics)
